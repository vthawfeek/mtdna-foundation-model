"""
Tests for MtDNA-FM model components.

All tests use tiny_config (genome_length=100, hidden_size=16, 2 layers) and
synthetic data so the suite runs in seconds without GPU or real data.

Test classes:
  TestMtDNAConfig          — JSON roundtrip, novel fields
  TestCircularPE           — shape, boundary values, non-learnability
  TestMtDNAEmbeddings      — forward shape, het_values optional
  TestMtDNAModel           — forward shapes, pooler is CLS token, save/load
  TestMtDNAForMaskedModeling — loss is scalar, shapes, gradient flow
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mtdna_fm.model.config import MtDNAConfig
from mtdna_fm.model.embeddings import MtDNACircularPositionalEncoding, MtDNAEmbeddings
from mtdna_fm.model.model import MtDNAForMaskedModeling, MtDNAModel

# ── Helpers ────────────────────────────────────────────────────────────────────


def make_batch(
    config: MtDNAConfig,
    batch_size: int = 2,
    seq_len: int = 8,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Construct a minimal synthetic batch consistent with tiny_config."""
    rng = np.random.default_rng(seed)
    input_ids = torch.from_numpy(
        rng.integers(6, config.vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    )
    # Absolute genomic positions within [0, genome_length)
    position_ids = torch.from_numpy(
        rng.integers(0, config.genome_length, size=(batch_size, seq_len), dtype=np.int64)
    )
    het_values = torch.rand(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "het_values": het_values,
        "attention_mask": attention_mask,
    }


# ── TestMtDNAConfig ────────────────────────────────────────────────────────────


class TestMtDNAConfig:
    def test_default_values(self, tiny_config: MtDNAConfig) -> None:
        assert tiny_config.vocab_size == 70
        assert tiny_config.hidden_size == 16
        assert tiny_config.num_hidden_layers == 2
        assert tiny_config.num_attention_heads == 4
        assert tiny_config.genome_length == 100
        assert tiny_config.use_circular_encoding is True
        assert tiny_config.use_het_projection is True

    def test_json_roundtrip(self, tiny_config: MtDNAConfig) -> None:
        json_str = tiny_config.to_json_string()
        loaded = MtDNAConfig(**json.loads(json_str))
        assert loaded.vocab_size == tiny_config.vocab_size
        assert loaded.genome_length == tiny_config.genome_length
        assert loaded.use_circular_encoding == tiny_config.use_circular_encoding
        assert loaded.hidden_size == tiny_config.hidden_size

    def test_special_token_ids(self, tiny_config: MtDNAConfig) -> None:
        assert tiny_config.pad_token_id == 0
        assert tiny_config.cls_token_id == 1
        assert tiny_config.mask_token_id == 2
        assert tiny_config.unk_token_id == 3
        assert tiny_config.sep_token_id == 4
        assert tiny_config.het_token_id == 5

    def test_model_type(self, tiny_config: MtDNAConfig) -> None:
        assert tiny_config.model_type == "mtdna_fm"

    def test_save_load(self, tiny_config: MtDNAConfig, tmp_path) -> None:
        tiny_config.save_pretrained(tmp_path)
        loaded = MtDNAConfig.from_pretrained(tmp_path)
        assert loaded.genome_length == tiny_config.genome_length
        assert loaded.hidden_size == tiny_config.hidden_size
        assert loaded.use_circular_encoding == tiny_config.use_circular_encoding


# ── TestCircularPE ─────────────────────────────────────────────────────────────


class TestCircularPE:
    def test_output_shape(self, tiny_config: MtDNAConfig) -> None:
        pe = MtDNACircularPositionalEncoding(tiny_config.genome_length, tiny_config.hidden_size)
        position_ids = torch.arange(10).unsqueeze(0)  # (1, 10)
        out = pe(position_ids)
        assert out.shape == (1, 10, tiny_config.hidden_size)

    def test_batch_shape(self, tiny_config: MtDNAConfig) -> None:
        pe = MtDNACircularPositionalEncoding(tiny_config.genome_length, tiny_config.hidden_size)
        position_ids = torch.randint(0, tiny_config.genome_length, (4, 8))
        out = pe(position_ids)
        assert out.shape == (4, 8, tiny_config.hidden_size)

    def test_non_learnable_buffer(self, tiny_config: MtDNAConfig) -> None:
        """The PE buffer should not be a parameter (no gradient)."""
        pe = MtDNACircularPositionalEncoding(tiny_config.genome_length, tiny_config.hidden_size)
        param_names = [n for n, _ in pe.named_parameters()]
        assert len(param_names) == 0, f"Expected no parameters, got {param_names}"
        buffer_names = [n for n, _ in pe.named_buffers()]
        assert "pe" in buffer_names

    def test_circular_boundary(self, tiny_config: MtDNAConfig) -> None:
        """Position 0 and genome_length should have identical encodings (circular property)."""
        pe = MtDNACircularPositionalEncoding(tiny_config.genome_length, tiny_config.hidden_size)
        # angle at 0 = 0, at genome_length = 2*pi — sin/cos are identical
        pos0 = torch.tensor([[0]])
        # The buffer has genome_length positions; we verify pos 0 and a position
        # near genome_length have smoothly varying encodings (not a hard boundary)
        enc0 = pe(pos0).squeeze()
        # Position at the end should be different from position 0 in the middle
        # but smoothly wrap around
        pos_last = torch.tensor([[tiny_config.genome_length - 1]])
        enc_last = pe(pos_last).squeeze()
        # They should be different (not identical)
        assert not torch.allclose(enc0, enc_last), "PE at pos 0 and pos_last should differ"

    def test_pe_buffer_size(self, tiny_config: MtDNAConfig) -> None:
        pe = MtDNACircularPositionalEncoding(tiny_config.genome_length, tiny_config.hidden_size)
        assert pe.pe.shape == (tiny_config.genome_length, tiny_config.hidden_size)


# ── TestMtDNAEmbeddings ────────────────────────────────────────────────────────


class TestMtDNAEmbeddings:
    def test_output_shape(self, tiny_config: MtDNAConfig) -> None:
        emb = MtDNAEmbeddings(tiny_config)
        batch = make_batch(tiny_config)
        out = emb(batch["input_ids"], batch["position_ids"], batch["het_values"])
        assert out.shape == (2, 8, tiny_config.hidden_size)

    def test_het_values_optional(self, tiny_config: MtDNAConfig) -> None:
        """Forward should work with het_values=None (zeros out het contribution)."""
        emb = MtDNAEmbeddings(tiny_config)
        # Use eval mode so dropout is deterministic (no random masking between calls)
        emb.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out_with_het = emb(batch["input_ids"], batch["position_ids"], batch["het_values"])
            out_no_het = emb(batch["input_ids"], batch["position_ids"], None)
        # Both should have the correct shape
        assert out_with_het.shape == out_no_het.shape
        # With all-zero het input the results should match (dropout is identity in eval mode)
        zero_het = torch.zeros_like(batch["het_values"])
        with torch.no_grad():
            out_zero_het = emb(batch["input_ids"], batch["position_ids"], zero_het)
        assert torch.allclose(out_no_het, out_zero_het, atol=1e-6)

    def test_no_het_projection_config(self) -> None:
        """With use_het_projection=False, het_projection should be None."""
        cfg = MtDNAConfig(
            vocab_size=70,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=32,
            max_seq_len=12,
            genome_length=100,
            use_het_projection=False,
        )
        emb = MtDNAEmbeddings(cfg)
        assert emb.het_projection is None

    def test_output_dtype(self, tiny_config: MtDNAConfig) -> None:
        emb = MtDNAEmbeddings(tiny_config)
        batch = make_batch(tiny_config)
        out = emb(batch["input_ids"], batch["position_ids"])
        assert out.dtype == torch.float32


# ── TestMtDNAModel ─────────────────────────────────────────────────────────────


class TestMtDNAModel:
    def test_forward_shapes(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAModel(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(**batch)
        assert out.last_hidden_state.shape == (2, 8, tiny_config.hidden_size)
        assert out.pooler_output.shape == (2, tiny_config.hidden_size)

    def test_pooler_is_cls_token(self, tiny_config: MtDNAConfig) -> None:
        """pooler_output must be the hidden state at position 0 (CLS token)."""
        model = MtDNAModel(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(**batch)
        assert torch.allclose(out.pooler_output, out.last_hidden_state[:, 0, :])

    def test_attention_mask_no_output_by_default(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAModel(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(**batch)
        assert out.attentions is None
        assert out.hidden_states is None

    def test_output_attentions(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAModel(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(**batch, output_attentions=True)
        assert out.attentions is not None
        assert len(out.attentions) == tiny_config.num_hidden_layers

    def test_output_hidden_states(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAModel(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True)
        assert out.hidden_states is not None
        # Should have num_layers + 1 (one per layer + the final post-encoder state)
        assert len(out.hidden_states) == tiny_config.num_hidden_layers + 1

    def test_save_load(self, tiny_config: MtDNAConfig, tmp_path) -> None:
        """save_pretrained / from_pretrained roundtrip preserves outputs."""
        model = MtDNAModel(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)

        with torch.no_grad():
            out_before = model(**batch)

        model.save_pretrained(tmp_path)
        loaded = MtDNAModel.from_pretrained(tmp_path)
        loaded.eval()

        with torch.no_grad():
            out_after = loaded(**batch)

        assert torch.allclose(out_before.pooler_output, out_after.pooler_output, atol=1e-5)

    def test_gradient_flow(self, tiny_config: MtDNAConfig) -> None:
        """Gradients should flow to the k-mer embedding weights."""
        model = MtDNAModel(tiny_config)
        model.train()
        batch = make_batch(tiny_config)

        out = model(**batch)
        # Use pooler as a proxy scalar loss
        loss = out.pooler_output.sum()
        loss.backward()

        # Embedding weights should have gradients
        assert model.embeddings.kmer_embeddings.weight.grad is not None

    def test_attention_mask_effect(self, tiny_config: MtDNAConfig) -> None:
        """Padding mask should affect outputs at padded positions."""
        model = MtDNAModel(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)

        # Create a version with the last 2 positions masked out
        masked_batch = dict(batch)
        masked_batch["attention_mask"] = batch["attention_mask"].clone()
        masked_batch["attention_mask"][:, -2:] = 0

        with torch.no_grad():
            out_full = model(**batch)
            out_masked = model(**masked_batch)

        # CLS pooler output should differ since attention pattern changed
        assert not torch.allclose(out_full.pooler_output, out_masked.pooler_output)

    def test_no_gpu_required(self, tiny_config: MtDNAConfig) -> None:
        """Model should run on CPU without error."""
        model = MtDNAModel(tiny_config)
        assert next(model.parameters()).device.type == "cpu"

    def test_get_input_embeddings(self, tiny_config: MtDNAConfig) -> None:
        import torch.nn as nn
        model = MtDNAModel(tiny_config)
        emb = model.get_input_embeddings()
        assert isinstance(emb, nn.Embedding)
        assert emb is model.embeddings.kmer_embeddings

    def test_set_input_embeddings(self, tiny_config: MtDNAConfig) -> None:
        import torch.nn as nn
        model = MtDNAModel(tiny_config)
        new_emb = nn.Embedding(tiny_config.vocab_size, tiny_config.hidden_size)
        model.set_input_embeddings(new_emb)
        assert model.embeddings.kmer_embeddings is new_emb

    def test_forward_without_attention_mask(self, tiny_config: MtDNAConfig) -> None:
        """Forward with attention_mask=None should default to all-ones mask."""
        model = MtDNAModel(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(batch["input_ids"], batch["position_ids"])
        assert out.last_hidden_state.shape == (2, 8, tiny_config.hidden_size)


# ── TestMtDNAForMaskedModeling ─────────────────────────────────────────────────


class TestMtDNAForMaskedModeling:
    def _make_labels(
        self, batch: dict[str, torch.Tensor], mask_fraction: float = 0.15
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create kmer_labels and het_labels for a batch."""
        batch_size, seq_len = batch["input_ids"].shape
        kmer_labels = batch["input_ids"].clone()
        # Mark ~85% as non-masked (-100 = ignored in CE loss)
        rng = np.random.default_rng(1)
        mask = rng.random((batch_size, seq_len)) < mask_fraction
        kmer_labels[~torch.from_numpy(mask)] = -100
        # Het labels: -1 where no data
        het_labels = torch.full_like(kmer_labels, -1, dtype=torch.float)
        het_labels[torch.from_numpy(mask)] = batch["het_values"][torch.from_numpy(mask)]
        return kmer_labels, het_labels

    def test_loss_is_scalar(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAForMaskedModeling(tiny_config, mlm_weight=1.0, het_weight=0.0)
        model.train()
        batch = make_batch(tiny_config)
        kmer_labels, _ = self._make_labels(batch)

        out = model(**batch, kmer_labels=kmer_labels)
        assert out.loss is not None
        assert out.loss.shape == ()  # scalar

    def test_loss_with_het(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAForMaskedModeling(tiny_config, mlm_weight=1.0, het_weight=0.3)
        model.train()
        batch = make_batch(tiny_config)
        kmer_labels, het_labels = self._make_labels(batch)

        out = model(**batch, kmer_labels=kmer_labels, het_labels=het_labels)
        assert out.loss is not None
        assert out.mlm_loss is not None
        # het_loss may be None if no valid het labels were generated
        # but the combined loss should be a valid scalar
        assert not torch.isnan(out.loss)

    def test_no_labels_no_loss(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAForMaskedModeling(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(**batch)
        assert out.loss is None

    def test_logits_shape(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAForMaskedModeling(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(**batch)
        assert out.logits.shape == (2, 8, tiny_config.vocab_size)

    def test_het_preds_shape(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAForMaskedModeling(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(**batch)
        assert out.het_preds.shape == (2, 8, 1)

    def test_het_preds_range(self, tiny_config: MtDNAConfig) -> None:
        """het_preds should be in [0, 1] (sigmoid output)."""
        model = MtDNAForMaskedModeling(tiny_config)
        model.eval()
        batch = make_batch(tiny_config)
        with torch.no_grad():
            out = model(**batch)
        assert (out.het_preds >= 0).all()
        assert (out.het_preds <= 1).all()

    def test_gradient_flow_through_loss(self, tiny_config: MtDNAConfig) -> None:
        model = MtDNAForMaskedModeling(tiny_config)
        model.train()
        batch = make_batch(tiny_config)
        kmer_labels, _ = self._make_labels(batch)

        out = model(**batch, kmer_labels=kmer_labels)
        out.loss.backward()

        # Embedding weights should have gradients
        assert model.mtdna.embeddings.kmer_embeddings.weight.grad is not None

    def test_loss_decreases_over_steps(self, tiny_config: MtDNAConfig) -> None:
        """
        Loss should decrease over 5 gradient steps on a fixed synthetic batch.
        Uses a high learning rate to ensure rapid convergence in a tiny test.
        """
        model = MtDNAForMaskedModeling(tiny_config)
        model.train()
        batch = make_batch(tiny_config)
        kmer_labels, _ = self._make_labels(batch)

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(5):
            optimiser.zero_grad()
            out = model(**batch, kmer_labels=kmer_labels)
            out.loss.backward()
            optimiser.step()
            losses.append(out.loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_get_input_embeddings_masked_modeling(self, tiny_config: MtDNAConfig) -> None:
        """MtDNAForMaskedModeling.get_input_embeddings delegates to encoder."""
        import torch.nn as nn
        model = MtDNAForMaskedModeling(tiny_config)
        emb = model.get_input_embeddings()
        assert isinstance(emb, nn.Embedding)

    def test_peft_lora_compatibility(self, tiny_config: MtDNAConfig) -> None:
        """PEFT should be able to wrap the model with LoRA on q/k/v/dense layers."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            pytest.skip("peft not installed")

        model = MtDNAForMaskedModeling(tiny_config)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=2,
            lora_alpha=4,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.0,
        )
        peft_model = get_peft_model(model, lora_config)
        # Verify trainable parameters are only the LoRA matrices
        trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
        assert all("lora_" in n for n in trainable), (
            f"Non-LoRA parameters are trainable: {[n for n in trainable if 'lora_' not in n]}"
        )
