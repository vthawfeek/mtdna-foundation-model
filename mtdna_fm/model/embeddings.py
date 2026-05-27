"""
Input embedding layer for mtDNA-FM.

Each k-mer token at genomic position i contributes three additive components:

1. kmer_embeddings (nn.Embedding):
   Maps discrete k-mer token ID → dense vector of shape (hidden_size,).
   Analogous to word embeddings in BERT. After training, similar k-mers
   (frequent co-occurrences, conserved functional sites) will have similar vectors.

2. MtDNACircularPositionalEncoding (fixed buffer, not learnable):
   Maps absolute genomic position → (hidden_size,) using a circular sinusoidal
   formula. This is the key novel component.

   WHY circular (not standard sinusoidal):
     Standard BERT PE uses sin/cos of pos / 10000^(2i/d). At positions 0 and
     16568, these values are maximally different — the encoding treats them as
     the most distant positions possible. But genomically, position 16568 is
     adjacent to position 0 because mtDNA is circular. The correct encoding
     uses 2π * pos / genome_length as the base angle, so the encoding at pos=0
     and pos=16569 are identical: sin(2π) = sin(0). The circular topology is
     encoded as a mathematical fact, not a learned approximation.

   WHY fixed (not learnable):
     The circular topology is not a statistical property of the training data —
     it is a biological fact about mitochondrial DNA. Fixing the encoding as a
     non-learnable buffer means it cannot be corrupted by gradient updates
     during fine-tuning or adapted away by a LoRA adapter. It is also
     interpretable: each position's encoding is deterministic and can be
     computed analytically.

   Formula:
     angle[pos] = 2π * pos / genome_length
     PE[pos, 2i]   = sin(angle[pos] * 1/10000^(2i/d))
     PE[pos, 2i+1] = cos(angle[pos] * 1/10000^(2i/d))

3. het_projection (Linear + LayerNorm, optional):
   Maps scalar heteroplasmy level (0.0 to 1.0) → (hidden_size,) and adds
   to the combined embedding.
   WHY: The same k-mer at the same position can have heteroplasmy 0.0 (purely
   wild-type) or 0.9 (almost entirely mutant). A learned linear transformation
   of the scalar heteroplasmy value allows the model to distinguish these cases.
   LayerNorm stabilises the projection output to have similar scale as the
   k-mer and positional embeddings.
   WHY optional (not always on): Phase 1 cross-species pre-training has no
   heteroplasmy data. The projection zeros out gracefully when het_values
   are all-zero tensors.

Final output: Dropout(LayerNorm(kmer_emb + circular_pe + het_proj))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mtdna_fm.model.config import MtDNAConfig


class MtDNACircularPositionalEncoding(nn.Module):
    """
    Fixed circular sinusoidal positional encoding for mtDNA's circular genome.

    Pre-computes all genome_length position encodings at init time and
    stores them as a non-learnable buffer. The forward pass indexes into
    this buffer using absolute genomic position IDs.

    This module is domain-agnostic: pass a different genome_length to use
    it for any circular nucleic acid (plasmid, viral genome, etc.).
    """

    def __init__(self, genome_length: int, hidden_size: int) -> None:
        super().__init__()
        pe = torch.zeros(genome_length, hidden_size)
        position = torch.arange(genome_length).float()

        # Circular angle: 2*pi * pos / genome_length
        # At pos=0 and pos=genome_length the angle is 0 and 2*pi respectively,
        # so sin/cos are identical — the junction is encoded smoothly.
        angle = 2 * math.pi * position / genome_length  # (genome_length,)

        # div_term shape: (hidden_size // 2,)
        # This is the standard 10000-based frequency scaling from "Attention Is All You Need"
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )

        # Outer product: (genome_length, 1) * (1, hidden_size//2) → (genome_length, hidden_size//2)
        pe[:, 0::2] = torch.sin(angle.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(angle.unsqueeze(1) * div_term)

        # Register as buffer: saved with model state, not a parameter (no gradient)
        self.register_buffer("pe", pe)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_ids: (batch, seq_len) absolute genomic coordinates [0, genome_length)

        Returns:
            (batch, seq_len, hidden_size) positional encodings
        """
        return self.pe[position_ids]  # type: ignore[index]


class MtDNAEmbeddings(nn.Module):
    """
    Combines k-mer identity, circular positional encoding, and optional
    heteroplasmy projection into a single hidden-size embedding vector per token.
    """

    def __init__(self, config: MtDNAConfig) -> None:
        super().__init__()

        # K-mer identity embedding
        self.kmer_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # Circular positional encoding (fixed buffer)
        if config.use_circular_encoding:
            self.circular_pe = MtDNACircularPositionalEncoding(
                genome_length=config.genome_length,
                hidden_size=config.hidden_size,
            )
        else:
            # Learnable positional embeddings as fallback (useful for ablation)
            self.circular_pe = nn.Embedding(config.max_seq_len, config.hidden_size)  # type: ignore[assignment]

        # Heteroplasmy projection: scalar [0, 1] → hidden_size
        if config.use_het_projection:
            self.het_projection = nn.Linear(1, config.hidden_size, bias=True)
            self.het_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.het_projection = None
            self.het_norm = None

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len)
        position_ids: torch.Tensor,  # (batch, seq_len) absolute genomic coords
        het_values: torch.Tensor | None = None,  # (batch, seq_len) float in [0, 1]
    ) -> torch.Tensor:  # (batch, seq_len, hidden_size)
        emb = self.kmer_embeddings(input_ids)
        emb = emb + self.circular_pe(position_ids)

        if self.het_projection is not None:
            if het_values is None:
                het_values = torch.zeros_like(input_ids, dtype=torch.float)
            # Unsqueeze to (batch, seq_len, 1) for the Linear layer
            het_proj = self.het_norm(self.het_projection(het_values.unsqueeze(-1)))
            emb = emb + het_proj

        return self.dropout(self.layer_norm(emb))
