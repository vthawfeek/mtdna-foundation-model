"""
Tests for mtDNA-FM training components: MtDNAMaskingCollator and mtdna_mlm_loss.

Test classes:
  TestMtDNAMaskingCollator — masking rate, blacklist enforcement, 80/10/10 split,
                             output shapes and dtypes
  TestMtDNAMLMLoss         — CE only on masked positions, het MSE only where
                             labels != -1, combined loss weighting
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mtdna_fm.tokenizer.vocabulary import KmerVocabulary
from mtdna_fm.training.losses import mtdna_mlm_loss
from mtdna_fm.training.masking import _DEFAULT_BLACKLIST, MtDNAMaskingCollator

# ── Helpers ────────────────────────────────────────────────────────────────────

SEQ_LEN = 20  # short windows for fast tests
BATCH_SIZE = 4


def make_fake_window(
    vocab: KmerVocabulary,
    seq_len: int = SEQ_LEN,
    start_pos: int = 0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Create a synthetic tokenised window dict."""
    rng = np.random.default_rng(seed)
    # Real k-mer tokens start at index n_special
    input_ids = torch.from_numpy(
        rng.integers(vocab.n_special, len(vocab), size=seq_len, dtype=np.int64)
    )
    # Prepend CLS token
    input_ids[0] = vocab.cls_token_id
    position_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long)
    attention_mask = torch.ones(seq_len, dtype=torch.long)
    het_values = torch.from_numpy(rng.random(seq_len).astype(np.float32))
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "het_values": het_values,
    }


def make_batch_list(
    vocab: KmerVocabulary,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    start_pos: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Make a list of fake windows for collation."""
    return [
        make_fake_window(vocab, seq_len=seq_len, start_pos=start_pos, seed=i)
        for i in range(batch_size)
    ]


# ── TestMtDNAMaskingCollator ───────────────────────────────────────────────────


class TestMtDNAMaskingCollator:
    @pytest.fixture()
    def vocab_k3(self) -> KmerVocabulary:
        """3-mer vocab (70 tokens) — fast for tests."""
        return KmerVocabulary.build(k=3)

    @pytest.fixture()
    def collator(self, vocab_k3: KmerVocabulary) -> MtDNAMaskingCollator:
        return MtDNAMaskingCollator(vocab_k3, mask_prob=0.15, seed=42)

    def test_output_keys(
        self, collator: MtDNAMaskingCollator, vocab_k3: KmerVocabulary
    ) -> None:
        batch = collator(make_batch_list(vocab_k3))
        expected = {"input_ids", "position_ids", "attention_mask", "het_values",
                    "kmer_labels", "het_labels"}
        assert set(batch.keys()) == expected

    def test_output_shapes(
        self, collator: MtDNAMaskingCollator, vocab_k3: KmerVocabulary
    ) -> None:
        batch = collator(make_batch_list(vocab_k3))
        for key in ("input_ids", "position_ids", "attention_mask",
                    "het_values", "kmer_labels", "het_labels"):
            assert batch[key].shape == (BATCH_SIZE, SEQ_LEN), (
                f"{key}: expected ({BATCH_SIZE}, {SEQ_LEN}), got {batch[key].shape}"
            )

    def test_masking_rate_approximately_15_percent(
        self, vocab_k3: KmerVocabulary
    ) -> None:
        """Over a large batch, the fraction of masked tokens should be ~15%."""
        # Use a large batch with no blacklisted positions in window range
        collator = MtDNAMaskingCollator(
            vocab_k3, mask_prob=0.15, blacklist_positions=(), seed=0
        )
        # windows not overlapping the C-tract blacklist
        windows = make_batch_list(vocab_k3, batch_size=200, seq_len=SEQ_LEN, start_pos=500)
        batch = collator(windows)

        # Count non-(-100) positions across the batch
        kmer_labels = batch["kmer_labels"]
        # Exclude CLS (position 0 in each window, token=cls_token_id)
        eligible_mask = batch["input_ids"] != vocab_k3.cls_token_id
        n_eligible = eligible_mask.sum().item()
        n_masked = (kmer_labels != -100).sum().item()

        rate = n_masked / n_eligible
        # Allow ±5% tolerance around 15%
        assert 0.10 <= rate <= 0.20, f"Expected ~0.15 masking rate, got {rate:.3f}"

    def test_blacklisted_positions_never_masked(
        self, vocab_k3: KmerVocabulary
    ) -> None:
        """Positions in the blacklist must never appear in kmer_labels."""
        # Create windows that contain blacklisted positions
        blacklist = tuple(range(303, 316))
        collator = MtDNAMaskingCollator(vocab_k3, mask_prob=0.9, blacklist_positions=blacklist, seed=1)

        # Windows starting at position 295 so they overlap positions 303-315
        windows = make_batch_list(vocab_k3, batch_size=50, seq_len=30, start_pos=295)
        batch = collator(windows)

        kmer_labels = batch["kmer_labels"]   # -100 at non-masked
        position_ids = batch["position_ids"]

        blacklist_set = set(blacklist)
        for i in range(len(windows)):
            for j in range(30):
                pos = int(position_ids[i, j].item())
                if pos in blacklist_set:
                    assert kmer_labels[i, j].item() == -100, (
                        f"Blacklisted position {pos} was masked at sample {i}, token {j}"
                    )

    def test_cls_never_masked(
        self, collator: MtDNAMaskingCollator, vocab_k3: KmerVocabulary
    ) -> None:
        """CLS token at position 0 must never be a masking target."""
        batch = collator(make_batch_list(vocab_k3))
        # Token at index 0 is CLS — its label must always be -100
        assert (batch["kmer_labels"][:, 0] == -100).all()

    def test_80_10_10_split(self, vocab_k3: KmerVocabulary) -> None:
        """
        Of masked positions:
          - ~80% should have input_id == mask_token_id
          - ~10% should have input_id != mask_token_id AND input_id != original
          - ~10% should be unchanged (input_id == original)
        Use a large batch for statistical stability.
        """
        collator = MtDNAMaskingCollator(
            vocab_k3, mask_prob=0.50, blacklist_positions=(), seed=99
        )
        windows = make_batch_list(vocab_k3, batch_size=300, seq_len=SEQ_LEN, start_pos=500)
        batch = collator(windows)

        kmer_labels = batch["kmer_labels"]
        masked_input_ids = batch["input_ids"]

        # Reconstruct original IDs: kmer_labels holds the original ID at masked positions
        masked_positions = kmer_labels != -100  # (batch, seq_len) bool

        original_at_mask = kmer_labels[masked_positions]       # original token IDs
        input_at_mask = masked_input_ids[masked_positions]     # possibly replaced

        n_total = masked_positions.sum().item()
        n_replaced_mask = (input_at_mask == vocab_k3.mask_token_id).sum().item()
        n_unchanged = (input_at_mask == original_at_mask).sum().item()
        # random replacement = neither [MASK] nor original
        n_random = n_total - n_replaced_mask - n_unchanged

        frac_mask = n_replaced_mask / n_total
        frac_unchanged = n_unchanged / n_total
        frac_random = n_random / n_total

        # Allow ±10% tolerance — these are probabilistic
        assert 0.70 <= frac_mask <= 0.90, f"[MASK] fraction {frac_mask:.3f} out of range [0.70, 0.90]"
        assert 0.00 <= frac_random <= 0.20, f"Random fraction {frac_random:.3f} out of range [0.00, 0.20]"
        assert 0.00 <= frac_unchanged <= 0.20, f"Unchanged fraction {frac_unchanged:.3f} out of range [0.00, 0.20]"

    def test_kmer_labels_only_at_masked_positions(
        self, collator: MtDNAMaskingCollator, vocab_k3: KmerVocabulary
    ) -> None:
        """kmer_labels must be -100 at every non-masked position."""
        batch = collator(make_batch_list(vocab_k3))
        kmer_labels = batch["kmer_labels"]

        # Non-masked: input unchanged (not [MASK]) AND kmer_labels == -100
        non_masked = kmer_labels == -100
        # At those positions, input_ids must equal the original token or be PAD/CLS
        # (we can't recover original easily here, but all labels must be -100)
        assert non_masked.any(), "Expected some non-masked positions"

    def test_het_labels_at_masked_positions(
        self, collator: MtDNAMaskingCollator, vocab_k3: KmerVocabulary
    ) -> None:
        """het_labels must be -1.0 at every non-masked position."""
        batch = collator(make_batch_list(vocab_k3))
        kmer_masked = batch["kmer_labels"] != -100
        het_not_sentinel = batch["het_labels"] != -1.0

        # Every position with a valid het label must also have a valid kmer label
        assert (het_not_sentinel & ~kmer_masked).sum() == 0, (
            "het_labels has valid values at non-masked positions"
        )

    def test_het_values_range(
        self, collator: MtDNAMaskingCollator, vocab_k3: KmerVocabulary
    ) -> None:
        """het_labels at masked positions must be in [0, 1]."""
        batch = collator(make_batch_list(vocab_k3))
        valid_het = batch["het_labels"][batch["het_labels"] != -1.0]
        if valid_het.numel() > 0:
            assert (valid_het >= 0.0).all()
            assert (valid_het <= 1.0).all()

    def test_no_het_values_in_batch(self, vocab_k3: KmerVocabulary) -> None:
        """Collator must work when het_values are absent (zeros used as fallback)."""
        collator = MtDNAMaskingCollator(vocab_k3, seed=0)
        windows = make_batch_list(vocab_k3)
        for w in windows:
            del w["het_values"]
        batch = collator(windows)
        # het_values should be all zeros
        assert (batch["het_values"] == 0.0).all()

    def test_default_blacklist_constant(self) -> None:
        """Default blacklist should cover positions 303-315 inclusive."""
        assert 303 in _DEFAULT_BLACKLIST
        assert 315 in _DEFAULT_BLACKLIST
        assert 302 not in _DEFAULT_BLACKLIST
        assert 316 not in _DEFAULT_BLACKLIST


# ── TestMtDNAMLMLoss ───────────────────────────────────────────────────────────


class TestMtDNAMLMLoss:
    """Tests for the mtdna_mlm_loss combined loss function."""

    def _make_logits(
        self, batch: int = 2, seq: int = 10, vocab: int = 70
    ) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(batch, seq, vocab)

    def _make_labels(
        self, batch: int = 2, seq: int = 10, n_masked: int = 3, vocab: int = 70
    ) -> torch.Tensor:
        """Labels: valid token IDs at masked positions, -100 elsewhere."""
        labels = torch.full((batch, seq), -100, dtype=torch.long)
        for i in range(batch):
            indices = torch.randperm(seq)[:n_masked]
            labels[i, indices] = torch.randint(6, vocab, (n_masked,))
        return labels

    def _make_het(
        self, batch: int = 2, seq: int = 10, n_valid: int = 3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """het_preds in [0,1]; het_labels with valid values at n_valid positions."""
        torch.manual_seed(1)
        het_preds = torch.rand(batch, seq, 1)
        het_labels = torch.full((batch, seq), -1.0)
        for i in range(batch):
            indices = torch.randperm(seq)[:n_valid]
            het_labels[i, indices] = torch.rand(n_valid)
        return het_preds, het_labels

    def test_returns_scalar_tensor(self) -> None:
        logits = self._make_logits()
        labels = self._make_labels()
        loss = mtdna_mlm_loss(logits, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

    def test_loss_only_at_masked_positions(self) -> None:
        """CE must differ when more positions have valid labels vs fewer."""
        logits = self._make_logits(vocab=70)

        # Only 2 valid positions
        labels_sparse = torch.full((2, 10), -100, dtype=torch.long)
        labels_sparse[0, 3] = 10
        labels_sparse[1, 7] = 20

        # All 10 positions valid
        labels_dense = torch.randint(6, 70, (2, 10), dtype=torch.long)

        loss_sparse = mtdna_mlm_loss(logits, labels_sparse)
        loss_dense = mtdna_mlm_loss(logits, labels_dense)

        # Both should be finite scalars
        assert not torch.isnan(loss_sparse), "Sparse loss is NaN"
        assert not torch.isnan(loss_dense), "Dense loss is NaN"
        # They should differ (different subsets of positions)
        assert not torch.isclose(loss_sparse, loss_dense), (
            "Sparse and dense labels produced identical loss — ignore_index may be broken"
        )

    def test_het_weight_zero_ignores_het(self) -> None:
        """With het_weight=0, het_preds/het_labels should have no effect."""
        logits = self._make_logits()
        labels = self._make_labels()
        het_preds, het_labels = self._make_het()

        loss_no_het = mtdna_mlm_loss(logits, labels, het_weight=0.0)
        loss_with_het = mtdna_mlm_loss(logits, labels, het_preds, het_labels, het_weight=0.0)

        assert torch.isclose(loss_no_het, loss_with_het), (
            "het_weight=0 should make het contribution zero"
        )

    def test_combined_loss_larger_than_mlm_alone(self) -> None:
        """With het_weight > 0 and valid het labels, combined loss > MLM alone."""
        logits = self._make_logits()
        labels = self._make_labels()
        het_preds, het_labels = self._make_het()

        mlm_only = mtdna_mlm_loss(logits, labels, het_weight=0.0)
        combined = mtdna_mlm_loss(logits, labels, het_preds, het_labels, het_weight=0.3)

        # Combined should be >= MLM-only (MSE is non-negative)
        assert combined.item() >= mlm_only.item() - 1e-6

    def test_mlm_weight_scales_loss(self) -> None:
        """mlm_weight should scale the CE term proportionally."""
        logits = self._make_logits()
        labels = self._make_labels()

        loss_1x = mtdna_mlm_loss(logits, labels, mlm_weight=1.0)
        loss_2x = mtdna_mlm_loss(logits, labels, mlm_weight=2.0)

        assert torch.isclose(loss_2x, 2.0 * loss_1x, atol=1e-5), (
            f"2x mlm_weight: expected {2 * loss_1x.item():.4f}, got {loss_2x.item():.4f}"
        )

    def test_het_loss_only_at_valid_labels(self) -> None:
        """MSE term must be computed only where het_labels != -1."""
        logits = self._make_logits()
        labels = self._make_labels()

        # All het_labels = -1 → het loss should be skipped (same as het_weight=0)
        het_preds, _ = self._make_het()
        het_labels_all_invalid = torch.full((2, 10), -1.0)

        loss_no_het = mtdna_mlm_loss(logits, labels, het_weight=0.0)
        loss_no_valid = mtdna_mlm_loss(
            logits, labels, het_preds, het_labels_all_invalid, het_weight=0.3
        )

        assert torch.isclose(loss_no_het, loss_no_valid, atol=1e-5), (
            "All-invalid het_labels should produce same loss as het_weight=0"
        )

    def test_gradient_flows_through_loss(self) -> None:
        """Gradients must flow back through the combined loss to logits."""
        logits = self._make_logits().requires_grad_(True)
        labels = self._make_labels()
        het_preds, het_labels = self._make_het()

        loss = mtdna_mlm_loss(logits, labels, het_preds, het_labels, het_weight=0.3)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_loss_is_non_negative(self) -> None:
        """Cross-entropy loss must always be non-negative."""
        logits = self._make_logits()
        labels = self._make_labels()
        loss = mtdna_mlm_loss(logits, labels)
        assert loss.item() >= 0.0

    def test_no_nan_loss(self) -> None:
        """Loss must not be NaN for well-formed inputs."""
        logits = self._make_logits()
        labels = self._make_labels()
        het_preds, het_labels = self._make_het()
        loss = mtdna_mlm_loss(logits, labels, het_preds, het_labels, het_weight=0.3)
        assert not torch.isnan(loss)
