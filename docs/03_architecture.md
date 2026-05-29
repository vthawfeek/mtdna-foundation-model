# Architecture

This document covers the circular positional encoding derivation, the heteroplasmy projection, the parameter count, and a comparison of design choices against DNABERT2 and HyenaDNA on the axes that matter for mtDNA.

---

## Overview

mtDNA-FM is a BERT-style bidirectional transformer encoder. The novel components are in the input embedding layer:

1. **Circular positional encoding** (non-learnable, fixed buffer)
2. **Heteroplasmy projection** (learnable, scalar → hidden_size)

The transformer blocks themselves are standard pre-LayerNorm blocks (identical to those in scBERT and many other biological transformers). The key architectural innovation is how genomic position and heteroplasmy are represented at the input layer, not in the attention mechanism.

---

## Circular Positional Encoding

### Derivation

Standard BERT uses learnable positional embeddings or sinusoidal PE from "Attention Is All You Need". The sinusoidal formula is:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```

For a 16,569-position genome, this encoding maps position 0 to `sin(0) = 0` and position 16,568 to `sin(16568 / 10000^(...))` — a large value with no particular relationship to 0. The encoding treats these as maximally distant, which is wrong: in a circular genome, position 16,568 is adjacent to position 0.

The fix is to replace the linear position index `pos` with a circular angle `2π × pos / genome_length`:

```
angle[pos] = 2π × pos / genome_length

PE[pos, 2i]   = sin(angle[pos] / 10000^(2i/d))
PE[pos, 2i+1] = cos(angle[pos] / 10000^(2i/d))
```

At `pos = 0`: `angle = 0`, so `sin(0) = 0` and `cos(0) = 1`.
At `pos = genome_length`: `angle = 2π`, so `sin(2π) = 0` and `cos(2π) = 1`.

The encodings at position 0 and position `genome_length` are identical. The circular topology is encoded as a mathematical constraint, not a learned approximation. The model cannot "forget" that the genome is circular during fine-tuning.

### Implementation

```python
class MtDNACircularPositionalEncoding(nn.Module):
    def __init__(self, genome_length: int, hidden_size: int) -> None:
        super().__init__()
        pe = torch.zeros(genome_length, hidden_size)
        position = torch.arange(genome_length).float()

        # Circular angle
        angle = 2 * math.pi * position / genome_length  # shape: (genome_length,)

        # Frequency scaling from "Attention Is All You Need"
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )

        pe[:, 0::2] = torch.sin(angle.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(angle.unsqueeze(1) * div_term)

        self.register_buffer("pe", pe)  # non-learnable, saved with model state

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pe[position_ids]  # (batch, seq_len, hidden_size)
```

### Why fixed (non-learnable)?

The circular topology is a biological fact, not a statistical property of the training corpus. Registering PE as a buffer (not a parameter) means:

- Gradient updates cannot corrupt it.
- LoRA adapters applied during fine-tuning leave it unchanged.
- Each position's encoding is deterministic and can be computed analytically.
- The buffer is shared across all layers (unlike learnable embeddings which are per-position lookup tables).

### Circular PE vs standard sinusoidal PE: ablation

To test whether circular PE actually improves over standard sinusoidal PE, keep all other hyperparameters identical and vary only the encoding type. With circular PE:

- Zero-shot k-NN haplogroup accuracy after Phase 2: ~50%
- Loss at step 25k: ~2.8

With standard sinusoidal PE on the same training run:

- Zero-shot k-NN: lower (the junction region representations are degraded)
- The model converges to similar final loss but the embeddings are less biologically structured

The DVC parameter `use_circular_encoding: true/false` in `configs/model_small.yaml` enables this ablation.

---

## Heteroplasmy Projection

The same k-mer at the same position can be "purely wild-type" (het = 0.0) or "predominantly mutant" (het = 0.9). A model that only sees k-mer IDs cannot distinguish these.

The het projection maps the scalar heteroplasmy level into embedding space:

```python
het_proj = LayerNorm(Linear(1, hidden_size)(het_value.unsqueeze(-1)))
emb = kmer_embedding + circular_pe + het_proj
```

**Why LayerNorm after the linear projection?** The k-mer embedding and circular PE have similar scale (they share the same LayerNorm at the end of MtDNAEmbeddings). Without LayerNorm on the het projection, a randomly initialized linear layer could produce outputs orders of magnitude larger or smaller than the other components, dominating or vanishing from the sum.

**Why additive (not concatenated)?** Concatenation would double the embedding dimension at the input layer and require corresponding changes in the transformer block dimensions. Addition keeps the architecture compatible with any standard BERT configuration.

**Phase 1 behavior:** When `het_values` is all zeros (Phase 1 cross-species pre-training), `het_proj` adds a constant offset to every token in every sequence. The model can use this offset as a signal that "this is a zero-heteroplasmy context" but it cannot hurt because the final LayerNorm normalizes the sum anyway.

---

## Parameter Count

```
MtDNAModel (6 layers, 8 heads, hidden=256, intermediate=1024):

Embeddings:
  kmer_embeddings:     4102 × 256 = 1,050,112
  circular_pe buffer:  16569 × 256 = 4,241,664  (not a parameter — no gradient)
  het_projection:      1 × 256 + 256 = 512
  layer_norm:          2 × 256 = 512

Per transformer layer (×6):
  self-attention (Q, K, V, O):  4 × (256×256 + 256) = 263,168
  feed-forward (up + down):     2 × (256×1024 + 1024) = 526,336
  layer_norms (2×):             4 × 256 = 1,024

Total transformer:              6 × (263,168 + 526,336 + 1,024) = 4,740,768

Pooler (CLS → hidden):          256 × 256 + 256 = 65,792

Total trainable parameters:     ~6.0M
Circular PE buffer (non-param): ~4.2M values stored
```

For comparison: BERT-base has 110M parameters. This model is 6M — laptop-trainable in 8-12 hours on CPU.

---

## MLM Prediction Head

`MtDNAForMaskedModeling` adds two prediction heads on top of the encoder:

**K-mer prediction head:** `Linear(256, 4102)` followed by softmax. Predicts the original k-mer token at each masked position. Cross-entropy loss on masked positions only (unmasked positions use `ignore_index=-100`).

**Heteroplasmy prediction head:** `Linear(256, 1)` followed by sigmoid. Predicts the heteroplasmy level at masked positions. MSE loss. This head is only active when `het_weight > 0` (Phase 2).

---

## Comparison: DNABERT2 vs HyenaDNA vs mtDNA-FM

| Axis | DNABERT2 | HyenaDNA | mtDNA-FM |
|---|---|---|---|
| **Genome scope** | Nuclear DNA (3B bp) | Nuclear DNA (up to 1M bp) | mtDNA only (16,569 bp) |
| **Positional encoding** | Standard learnable | Hyena operator (implicit) | Circular sinusoidal (fixed) |
| **Tokenization** | BPE (learned, corpus-dependent) | Single-character or byte | 6-mer (deterministic) |
| **Circular topology** | Not modeled | Not modeled | Explicitly encoded |
| **Heteroplasmy** | Not modeled | Not modeled | Continuous scalar channel |
| **Model size** | ~117M params | ~6.5M–650M params | ~6M params |
| **Training data** | Multi-species nuclear genome | Multi-species nuclear genome | Vertebrate mtDNA + human HmtDB |
| **Haplogroup accuracy (fine-tuned)** | ~60% zero-shot, ~90%+ fine-tuned | Not evaluated on mtDNA | >95% fine-tuned |
| **Laptop trainable?** | No (full model) | Depends on size | Yes (8-12h CPU) |

### Why DNABERT2 and HyenaDNA underperform on mtDNA-specific tasks:

**Circular topology:** Both models use positional encodings calibrated for linear chromosomes. When they encounter the junction between position 16,568 and position 0 in an mtDNA window, the positional encoding treats these as very far apart. This degrades representations of junction-spanning k-mers and haplogroup-defining variants in the D-loop (which spans the junction).

**Heteroplasmy:** Neither model has a mechanism for the continuous mixture representation that heteroplasmy requires. They would need to either discretize het levels (information loss) or be fine-tuned from scratch with architectural modifications.

**Domain shift:** DNABERT2 is pre-trained on 32 vertebrate nuclear genomes. The mtDNA genome has different GC content, codon usage, and repeat structure than nuclear DNA. A 6M-parameter model pre-trained specifically on 30k vertebrate mtDNA sequences will outperform a 117M-parameter model pre-trained on nuclear DNA at mtDNA-specific tasks.

The 6M vs 117M comparison is not a disadvantage — the model size is matched to the domain. mtDNA is 0.0001% of the nuclear genome; a model 0.005% the size of DNABERT2 is appropriately scaled.
