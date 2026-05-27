"""
Transformer encoder layers for mtDNA-FM.

WHY pre-LayerNorm (not post-LayerNorm as in original BERT):
  Original BERT applies LayerNorm AFTER the residual addition:
    x = LayerNorm(x + Attention(x))
  Pre-norm applies it BEFORE:
    x = x + Attention(LayerNorm(x))
  Pre-norm provides more stable gradient flow, especially important when
  training with limited data or on a tight compute budget. It avoids the
  gradient vanishing problem that can occur in deep post-norm networks.
  Modern transformers (GPT-2, LLaMA, Geneformer) universally use pre-norm.

WHY bidirectional attention:
  Each k-mer token can attend to all other k-mers in the window simultaneously.
  For sequence-level tasks like haplogroup classification, the model needs to
  integrate signal from across the whole window — a variant at position i may
  have functional context established by positions i-200 through i+200. Causal
  (unidirectional) attention would prevent this.

Layer composition:
  MtDNALayer:
    x → pre_norm → MultiHeadAttention → + residual
    x → pre_norm → FFN (2-layer MLP) → + residual

  FFN (Feed-Forward Network):
    Linear(hidden → intermediate) → GELU → Linear(intermediate → hidden)
    intermediate_size = 4 × hidden_size follows the standard BERT ratio.
    GELU activation is smoother than ReLU and empirically better for
    transformer pretraining.

Named layers (query, key, value, dense) are intentional: PEFT's get_peft_model()
targets them by name when applying LoRA adapters. Renaming these breaks LoRA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtdna_fm.model.config import MtDNAConfig


class MtDNAAttention(nn.Module):
    """Multi-head self-attention with pre-LayerNorm."""

    def __init__(self, config: MtDNAConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        # Named query/key/value/dense so PEFT can target them by name for LoRA
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # output projection

        self.attn_dropout = nn.Dropout(config.attention_dropout_prob)
        self.proj_dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch, seq_len, hidden_size)
        attention_mask: torch.Tensor | None = None,  # (batch, 1, 1, seq_len) additive
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)  # pre-norm

        batch, seq_len, _ = hidden_states.shape

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            # (batch, seq_len, hidden) → (batch, heads, seq_len, head_dim)
            return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.query(hidden_states))
        k = split_heads(self.key(hidden_states))
        v = split_heads(self.value(hidden_states))

        scale = self.head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch, heads, seq, seq)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        output = self.proj_dropout(self.dense(context))

        return residual + output, attn_weights if output_attentions else None


class MtDNAFFN(nn.Module):
    """Position-wise feed-forward network with pre-LayerNorm and GELU activation."""

    def __init__(self, config: MtDNAConfig) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)  # pre-norm
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.dropout(self.fc2(hidden_states))
        return residual + hidden_states


class MtDNALayer(nn.Module):
    """Single mtDNA-FM transformer encoder layer: attention + FFN."""

    def __init__(self, config: MtDNAConfig) -> None:
        super().__init__()
        self.attention = MtDNAAttention(config)
        self.ffn = MtDNAFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states, attn_weights = self.attention(
            hidden_states, attention_mask, output_attentions
        )
        hidden_states = self.ffn(hidden_states)
        return hidden_states, attn_weights


class MtDNAEncoder(nn.Module):
    """Stack of N MtDNALayers."""

    def __init__(self, config: MtDNAConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MtDNALayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple | None, tuple | None]:
        all_hidden_states: list[torch.Tensor] | None = [] if output_hidden_states else None
        all_attentions: list[torch.Tensor] | None = [] if output_attentions else None

        for layer in self.layers:
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states, attn_weights = layer(hidden_states, attention_mask, output_attentions)
            if output_attentions and attn_weights is not None and all_attentions is not None:
                all_attentions.append(attn_weights)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return (
            hidden_states,
            tuple(all_hidden_states) if all_hidden_states is not None else None,
            tuple(all_attentions) if all_attentions is not None else None,
        )
