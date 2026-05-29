"""
mtDNA-FM model classes: base encoder and masked-modeling pretraining wrapper.

WHY two separate classes (MtDNAModel vs MtDNAForMaskedModeling):
  This follows the standard HuggingFace pattern (BertModel / BertForMaskedLM).
  - MtDNAModel: the base encoder producing contextual embeddings. This is what
    gets saved, fine-tuned, and distributed. It has no pretraining-specific
    prediction heads.
  - MtDNAForMaskedModeling: adds k-mer identity and heteroplasmy prediction
    heads on top, used ONLY during pretraining. After pretraining, these heads
    are discarded and only MtDNAModel weights are kept for downstream use.
  This clean separation ensures the fine-tuning interface is clean and that
  no pretraining cruft leaks into downstream applications.

WHY inherit PreTrainedModel:
  - save_pretrained() / from_pretrained(): standardised weight I/O with safetensors
  - gradient_checkpointing_enable(): trades compute for memory on a laptop
  - PEFT/LoRA: get_peft_model() introspects PreTrainedModel to find Linear
    layers by name (query, key, value, dense) and wraps them automatically
  - Automatic device placement (.to(device), .cuda(), etc.)
  - HuggingFace Hub: push_to_hub() for sharing

Prediction heads in MtDNAForMaskedModeling:
  kmer_prediction_head: 3-layer MLP (hidden → hidden → vocab_size)
    The intermediate GELU layer gives the head enough capacity to map from
    contextual embeddings to k-mer identity without it being a bottleneck.
  het_prediction_head: Linear(hidden → 1) with sigmoid
    Heteroplasmy is bounded [0, 1], so sigmoid output is appropriate.
    Trained with MSE/Huber loss against observed heteroplasmy levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from mtdna_fm.model.config import MtDNAConfig
from mtdna_fm.model.embeddings import MtDNAEmbeddings
from mtdna_fm.model.transformer import MtDNAEncoder

# ── Output dataclasses ─────────────────────────────────────────────────────────

@dataclass
class MtDNAModelOutput(ModelOutput):
    """Outputs from MtDNAModel.forward()."""

    last_hidden_state: torch.Tensor | None = None
    pooler_output: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class MtDNAMaskedModelingOutput(ModelOutput):
    """Outputs from MtDNAForMaskedModeling.forward()."""

    loss: torch.Tensor | None = None
    mlm_loss: torch.Tensor | None = None
    het_loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None  # (batch, seq_len, vocab_size)
    het_preds: torch.Tensor | None = None  # (batch, seq_len, 1)
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


# ── Base encoder model ─────────────────────────────────────────────────────────


class MtDNAModel(PreTrainedModel):
    """
    Encoder-only mtDNA foundation model.

    Processes a windowed k-mer sequence through a bidirectional transformer
    with circular positional encoding and optional heteroplasmy channel.
    Returns contextual embeddings for each k-mer token plus a pooled genome
    embedding (CLS token at position 0).

    This is the base model for all downstream tasks. Instantiated during
    pretraining and saved at the end; the prediction heads from
    MtDNAForMaskedModeling are discarded.
    """

    config_class = MtDNAConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MtDNAConfig) -> None:
        super().__init__(config)
        self.embeddings = MtDNAEmbeddings(config)
        self.encoder = MtDNAEncoder(config)
        # Final layer norm applied after the last encoder layer
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Required by PreTrainedModel for gradient checkpointing."""
        return self.embeddings.kmer_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.kmer_embeddings = value

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, MtDNAEncoder):
            module.gradient_checkpointing = value

    def _make_additive_attention_mask(
        self,
        attention_mask: torch.Tensor,  # (batch, seq_len) int, 1=attend 0=ignore
    ) -> torch.Tensor:
        """
        Convert a binary attention mask to an additive mask for attention scores.
        Attended positions → 0.0, ignored (PAD) positions → -10000.0.
        This is added to raw attention scores before softmax, driving PAD
        attention weights to ~0.
        """
        # Expand to (batch, 1, 1, seq_len) for broadcasting across heads and query positions
        mask = attention_mask[:, None, None, :].float()
        mask = (1.0 - mask) * -10000.0
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len)
        position_ids: torch.Tensor,  # (batch, seq_len) absolute genomic coords
        het_values: torch.Tensor | None = None,  # (batch, seq_len) float in [0, 1]
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> MtDNAModelOutput:

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        additive_mask = self._make_additive_attention_mask(attention_mask)

        hidden_states = self.embeddings(input_ids, position_ids, het_values)

        hidden_states, all_hidden, all_attentions = self.encoder(
            hidden_states,
            attention_mask=additive_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = self.final_layer_norm(hidden_states)

        # Genome-level embedding = CLS token representation at position 0
        pooler_output = hidden_states[:, 0, :]

        return MtDNAModelOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            hidden_states=all_hidden,
            attentions=all_attentions,
        )


# ── Pretraining model ──────────────────────────────────────────────────────────


class MtDNAForMaskedModeling(PreTrainedModel):
    """
    MtDNA-FM with masked k-mer prediction heads for pretraining.

    Adds on top of MtDNAModel:
      kmer_prediction_head:  3-layer MLP (hidden → hidden → vocab_size) for k-mer identity (CE loss)
      het_prediction_head:   Linear(hidden → 1) + sigmoid for heteroplasmy level (MSE loss)

    Combined loss:
      total = mlm_weight * CE(kmer_logits, kmer_labels)
            + het_weight * MSE(het_preds, het_labels)
    Only computed at masked positions (kmer_labels != -100).
    Het loss only computed where het_labels != -1 (i.e., where het data exists).

    After pretraining: discard this class and use MtDNAModel.from_pretrained()
    which loads only the encoder weights.
    """

    config_class = MtDNAConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: MtDNAConfig,
        mlm_weight: float = 1.0,
        het_weight: float = 0.0,
    ) -> None:
        super().__init__(config)
        self.mlm_weight = mlm_weight
        self.het_weight = het_weight

        self.mtdna = MtDNAModel(config)

        # K-mer identity prediction head
        # Two-layer MLP with intermediate GELU gives enough capacity without
        # making the head a bottleneck at 4,102-class output.
        self.kmer_prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        )

        # Heteroplasmy level prediction head
        # Sigmoid bounds output to [0, 1] to match the biological range.
        self.het_prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.mtdna.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        het_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kmer_labels: torch.Tensor | None = None,  # (batch, seq_len), -100 at non-masked
        het_labels: torch.Tensor | None = None,  # (batch, seq_len), -1 where no het data
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> MtDNAMaskedModelingOutput:

        encoder_output = self.mtdna(
            input_ids=input_ids,
            position_ids=position_ids,
            het_values=het_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden = encoder_output.last_hidden_state  # (batch, seq_len, hidden)
        kmer_logits = self.kmer_prediction_head(hidden)  # (batch, seq_len, vocab_size)
        het_preds = self.het_prediction_head(hidden)  # (batch, seq_len, 1)

        loss = mlm_loss = het_loss = None

        if kmer_labels is not None:
            mlm_loss = F.cross_entropy(
                kmer_logits.view(-1, kmer_logits.size(-1)),
                kmer_labels.view(-1),
                ignore_index=-100,
            )
            loss = self.mlm_weight * mlm_loss

            if self.het_weight > 0 and het_labels is not None:
                # Only compute het loss at positions with valid het labels
                valid = het_labels != -1
                if valid.any():
                    het_loss = F.mse_loss(
                        het_preds.squeeze(-1)[valid],
                        het_labels[valid].float(),
                    )
                    loss = loss + self.het_weight * het_loss

        return MtDNAMaskedModelingOutput(
            loss=loss,
            mlm_loss=mlm_loss,
            het_loss=het_loss,
            logits=kmer_logits,
            het_preds=het_preds,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


# ── Haplogroup classification model ───────────────────────────────────────────


@dataclass
class HaplogroupClassificationOutput(ModelOutput):
    """Outputs from MtDNAForHaplogroupClassification.forward()."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None  # (batch, num_labels)
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class MtDNAForHaplogroupClassification(PreTrainedModel):
    """
    Fine-tuning wrapper for haplogroup classification.

    Architecture: MtDNAModel encoder + Linear(hidden_size, num_labels) head.

    Input modes:
      Single window:   input_ids shape (batch, seq_len)
      Multiple windows: input_ids shape (batch, n_windows, seq_len)
        → CLS tokens are extracted per window and mean-pooled across windows,
          giving one genome-level embedding per sample before the classifier.
        This is the recommended path for fine-tuning on full mtDNA genomes,
        where each genome is split into ~63 overlapping 512-token windows.

    LoRA: apply PEFT get_peft_model() after construction with
      r=8, lora_alpha=16, target_modules=["query","key","value","dense"].
    """

    config_class = MtDNAConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        base_model: MtDNAModel,
        num_labels: int = 26,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(base_model.config)
        self.num_labels = num_labels
        self.mtdna = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
        self.post_init()

    def freeze_encoder(self) -> None:
        """Freeze encoder; only the classifier head updates."""
        for param in self.mtdna.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze for full fine-tuning."""
        for param in self.mtdna.parameters():
            param.requires_grad = True

    def get_input_embeddings(self) -> nn.Embedding:
        return self.mtdna.get_input_embeddings()

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, MtDNAModel):
            module._set_gradient_checkpointing(module.encoder, value)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        het_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> HaplogroupClassificationOutput:
        """
        Parameters
        ----------
        input_ids:
            (batch, seq_len) for single-window mode, or
            (batch, n_windows, seq_len) for multi-window mode.
        position_ids:
            Same shape as input_ids — absolute genomic coordinates.
        labels:
            (batch,) integer class indices in [0, num_labels).
        """
        multi_window = input_ids.dim() == 3

        if multi_window:
            batch_size, n_windows, seq_len = input_ids.shape
            # Flatten windows into the batch dimension for a single encoder pass
            flat_ids = input_ids.view(batch_size * n_windows, seq_len)
            flat_pos = position_ids.view(batch_size * n_windows, seq_len)
            flat_het = (
                het_values.view(batch_size * n_windows, seq_len)
                if het_values is not None
                else None
            )
            flat_mask = (
                attention_mask.view(batch_size * n_windows, seq_len)
                if attention_mask is not None
                else None
            )

            enc = self.mtdna(
                input_ids=flat_ids,
                position_ids=flat_pos,
                het_values=flat_het,
                attention_mask=flat_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # CLS per window → (batch, n_windows, hidden) → mean over windows
            cls_per_window = enc.pooler_output.view(batch_size, n_windows, -1)
            pooled = cls_per_window.mean(dim=1)  # (batch, hidden)
            all_hidden = enc.hidden_states
            all_attentions = enc.attentions
        else:
            enc = self.mtdna(
                input_ids=input_ids,
                position_ids=position_ids,
                het_values=het_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            pooled = enc.pooler_output  # (batch, hidden)
            all_hidden = enc.hidden_states
            all_attentions = enc.attentions

        logits = self.classifier(self.dropout(pooled))  # (batch, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return HaplogroupClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden,
            attentions=all_attentions,
        )


# ── Variant pathogenicity model ───────────────────────────────────────────────


@dataclass
class VariantPathogenicityOutput(ModelOutput):
    """Outputs from MtDNAForVariantPathogenicity.forward()."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None  # (batch,) raw BCE logits
    probs: torch.Tensor | None = None   # (batch,) sigmoid probabilities
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class MtDNAForVariantPathogenicity(PreTrainedModel):
    """
    Fine-tuning wrapper for binary variant pathogenicity prediction.

    Input: 512-token window centered on the variant position.
    Classification head reads the hidden state at the token containing the
    variant position — not CLS — because pathogenicity is a local property
    (effect on a specific codon, tRNA stem, or rRNA position), not a global
    genome property.

    Training data: ClinVar pathogenic variants (positive) vs gnomAD common
    variants AF > 0.01 (negative). pos_weight=2.5 in BCE loss compensates
    for the 1:2.5 positive-to-negative ratio.

    LoRA: r=4 (small dataset, heavier regularisation than haplogroup).
    """

    config_class = MtDNAConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        base_model: MtDNAModel,
        dropout: float = 0.1,
        pos_weight: float = 2.5,
    ) -> None:
        super().__init__(base_model.config)
        self.mtdna = base_model
        self.dropout = nn.Dropout(dropout)
        # Single linear head on the variant-position token hidden state
        self.classifier = nn.Linear(base_model.config.hidden_size, 1)
        # pos_weight stored as buffer so it moves to device with .to()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.post_init()

    def freeze_encoder(self) -> None:
        for param in self.mtdna.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.mtdna.parameters():
            param.requires_grad = True

    def get_input_embeddings(self) -> nn.Embedding:
        return self.mtdna.get_input_embeddings()

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, MtDNAModel):
            module._set_gradient_checkpointing(module.encoder, value)

    def forward(
        self,
        input_ids: torch.Tensor,       # (batch, seq_len)
        position_ids: torch.Tensor,    # (batch, seq_len) absolute genomic coords
        variant_token_idx: torch.Tensor,  # (batch,) index into seq_len for variant token
        het_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,  # (batch,) float 0/1
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> VariantPathogenicityOutput:
        """
        Parameters
        ----------
        variant_token_idx:
            Per-sample index pointing to the token in the window that contains
            the variant position. Used to index last_hidden_state to get the
            locally contextualised representation of the mutation site.
        labels:
            Float tensor of 0.0 (benign) or 1.0 (pathogenic).
        """
        enc = self.mtdna(
            input_ids=input_ids,
            position_ids=position_ids,
            het_values=het_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden = enc.last_hidden_state  # (batch, seq_len, hidden)

        # Gather the hidden state at the variant token for each sample
        # variant_token_idx: (batch,) → expand to (batch, 1, hidden)
        idx = variant_token_idx.clamp(0, hidden.size(1) - 1)
        variant_hidden = hidden[
            torch.arange(hidden.size(0), device=hidden.device), idx, :
        ]  # (batch, hidden)

        logits = self.classifier(self.dropout(variant_hidden)).squeeze(-1)  # (batch,)
        probs = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
                pos_weight=self.pos_weight,
            )

        return VariantPathogenicityOutput(
            loss=loss,
            logits=logits,
            probs=probs,
            hidden_states=enc.hidden_states,
            attentions=enc.attentions,
        )


# ── Heteroplasmy regression model ─────────────────────────────────────────────


@dataclass
class HeteroplasmyRegressionOutput(ModelOutput):
    """Outputs from MtDNAForHeteroplasmyRegression.forward()."""

    loss: torch.Tensor | None = None
    predictions: torch.Tensor | None = None  # (batch,) float in [0, 1]
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class MtDNAForHeteroplasmyRegression(PreTrainedModel):
    """
    Fine-tuning wrapper for heteroplasmy level regression.

    Predicts the mean observed heteroplasmy level (float in [0, 1]) for a
    variant from its local sequence context.

    Input: 512-token window centered on the variant position.
    The regression head reads the hidden state at the variant-position token
    (not CLS) — consistent with the pathogenicity model: heteroplasmy level
    is a local property of the variant's sequence context, not a global
    genome property.

    Head: Linear(hidden_size, 64) -> GELU -> Linear(64, 1) -> Sigmoid
    Sigmoid bounds the output to [0, 1] to match the biological range.

    Loss: Huber loss (delta=0.1) — more robust than MSE to the noise and
    outliers in gnomAD heteroplasmy level estimates, which are means over
    a small carrier population and can be noisy.

    Training data: gnomAD variants with >=50 heteroplasmic carriers.
    Target: mean observed heteroplasmy level (~1,000 data points).
    Evaluation: 5-fold cross-validation, R-squared and Spearman rho.

    LoRA: r=4 (small dataset, ~1,000 data points; heavier regularisation).
    """

    config_class = MtDNAConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        base_model: MtDNAModel,
        dropout: float = 0.1,
        huber_delta: float = 0.1,
    ) -> None:
        super().__init__(base_model.config)
        self.mtdna = base_model
        self.dropout = nn.Dropout(dropout)
        self.huber_delta = huber_delta
        # Head: Linear(hidden, 64) -> GELU -> Linear(64, 1) -> Sigmoid
        self.regression_head = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.post_init()

    def freeze_encoder(self) -> None:
        for param in self.mtdna.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.mtdna.parameters():
            param.requires_grad = True

    def get_input_embeddings(self) -> nn.Embedding:
        return self.mtdna.get_input_embeddings()

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, MtDNAModel):
            module._set_gradient_checkpointing(module.encoder, value)

    def forward(
        self,
        input_ids: torch.Tensor,          # (batch, seq_len)
        position_ids: torch.Tensor,       # (batch, seq_len)
        variant_token_idx: torch.Tensor,  # (batch,)
        het_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,  # (batch,) float in [0, 1]
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> HeteroplasmyRegressionOutput:
        """
        Parameters
        ----------
        variant_token_idx:
            Per-sample index pointing to the token that contains the variant
            position. Used to index last_hidden_state for locally contextualised
            variant representation.
        labels:
            Mean observed heteroplasmy level per variant, float in [0, 1].
        """
        enc = self.mtdna(
            input_ids=input_ids,
            position_ids=position_ids,
            het_values=het_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden = enc.last_hidden_state  # (batch, seq_len, hidden)

        idx = variant_token_idx.clamp(0, hidden.size(1) - 1)
        variant_hidden = hidden[
            torch.arange(hidden.size(0), device=hidden.device), idx, :
        ]  # (batch, hidden)

        predictions = self.regression_head(self.dropout(variant_hidden)).squeeze(-1)  # (batch,)

        loss = None
        if labels is not None:
            loss = F.huber_loss(predictions, labels.float(), delta=self.huber_delta)

        return HeteroplasmyRegressionOutput(
            loss=loss,
            predictions=predictions,
            hidden_states=enc.hidden_states,
            attentions=enc.attentions,
        )
