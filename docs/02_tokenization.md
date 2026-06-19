# Tokenization

This document covers 6-mer vocabulary construction, the circular windowing mechanic, the heteroplasmy channel, and vocabulary statistics.

---

## Why K-mer Tokenization?

DNA sequence models face a fundamental tokenization choice: character-level (4 tokens), k-mer (4^k tokens), or BPE (learned, corpus-dependent vocabulary).

**Character-level** keeps the vocabulary tiny but forces the model to learn multi-base patterns entirely through self-attention. Long-range dependencies in a 16,569-character sequence strain even a 512-token context window.

**BPE** learns which substrings are statistically frequent in the training corpus. This is excellent for natural language (where word-level units have meaning) but problematic for DNA: the learned vocabulary is non-reproducible across projects, depends on corpus composition, and discards the biological fact that every possible 6-mer is equally valid.

**K-mer tokenization** gives every possible 6-mer a stable, deterministic token ID. The vocabulary is 4^6 = 4,096 tokens plus 6 special tokens = 4,102 total. Any project using `KmerVocabulary.build(k=6)` on any machine will produce the same mapping. This is the right choice for a pre-trained model that needs to generalize across datasets without re-tokenizing.

---

## Vocabulary Construction

```python
from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

vocab = KmerVocabulary.build(k=6)
len(vocab)  # 4102
```

### Special tokens

The first 6 token IDs are reserved for special tokens:

| ID | Token | Purpose |
|---|---|---|
| 0 | `[PAD]` | Padding to fixed sequence length |
| 1 | `[CLS]` | Classification token prepended to every sequence window |
| 2 | `[MASK]` | Masked position during MLM pre-training |
| 3 | `[UNK]` | Any k-mer containing N (ambiguous base) |
| 4 | `[SEP]` | Separator (for future paired-sequence tasks) |
| 5 | `[HET]` | Heteroplasmic position marker (reserved for future use) |

### K-mer enumeration

K-mers are enumerated in lexicographic order over the alphabet ACGT. The index of a 6-mer is its position in sorted(all_4096_kmers), plus 6 (to leave room for special tokens). This ordering is deterministic and reproducible.

```python
vocab.encode("AAAAAA")  # 6 (first k-mer after special tokens)
vocab.encode("TTTTTT")  # 4101 (last k-mer)
vocab.decode(6)         # "AAAAAA"
```

N-containing k-mers (e.g., "ACGTAN") map to `[UNK]` (ID 3). This handles sequencing gaps without crashing.

### Save and load

The vocabulary follows HuggingFace `PretrainedConfig` conventions so it can be stored alongside model weights:

```python
vocab.save_pretrained("models/vocabulary/")
# writes: models/vocabulary/vocab_config.json

loaded = KmerVocabulary.from_pretrained("models/vocabulary/")
assert len(loaded) == 4102
```

---

## Sequence Tokenization

```python
from mtdna_fm.tokenizer.tokenize import tokenize_sequence

tokens = tokenize_sequence(
    seq="ATCG...",          # 16,569-bp mtDNA genome
    vocabulary=vocab,
    k=6,
    stride=1,
    max_seq_len=512,
    circular=True,
    het_levels=None,        # optional: np.ndarray of float in [0, 1]
)
# tokens: dict with keys input_ids, attention_mask, position_ids, het_values
```

### Output fields

| Field | Shape | Description |
|---|---|---|
| `input_ids` | `(seq_len,)` | K-mer token IDs |
| `attention_mask` | `(seq_len,)` | 1 for real tokens, 0 for padding |
| `position_ids` | `(seq_len,)` | Absolute genomic coordinates (0-indexed) |
| `het_values` | `(seq_len,)` | Heteroplasmy levels, 0.0 if not provided |

### Circular windowing

The full genome (16,569 bp) is too long for a 512-token context window. Instead, `MtDNADataset` tiles the genome with overlapping windows:

- Window size: 512 tokens
- Stride: 256 tokens (50% overlap)
- Windows per genome: ceil(16569 / 256) ≈ 65

Each window receives a `[CLS]` token prepended, so the actual context is 513 tokens. The `position_ids` in each window are **absolute genomic coordinates** (not window-relative), so the circular positional encoding maps each token to the correct angular position on the genome.

### Junction handling

With `circular=True`, tokenization wraps around the genome junction at position 16568/0. Before k-merizing, the last k-1 = 5 bases are appended to the front of the sequence:

```
seq_circular = seq[-5:] + seq  # 16,574 bp
```

This ensures that the k-mers at positions 16564-16568 (which overlap the junction) are computed correctly. Without this step, those positions would yield partial k-mers that don't appear in the vocabulary.

The `position_ids` for the wrapped junction tokens are assigned positions 16564-16568, not positions that exceed `genome_length`. The circular PE handles the topology.

---

## Heteroplasmy Channel

Heteroplasmy is the presence of two or more mitochondrial DNA variants within a single cell (e.g., 80% wild-type copies, 20% mutant copies). Standard sequence models expect one definitive base at each position; the heteroplasmy channel extends the model to handle continuous mixtures.

### Input format

`het_levels` is an optional `np.ndarray` of shape `(genome_length,)` with float values in `[0, 1]`. Each value is the fraction of mtDNA copies carrying an alternate allele at that position.

- `0.0`: all copies are wild-type at this position
- `0.5`: 50/50 mixture (maximum heteroplasmy)
- `1.0`: all copies carry the alternate allele (homoplasmic variant)

In most sequences, `het_levels` is all zeros. The model handles this gracefully: the het projection contributes zero to the embedding when all values are zero.

### How it feeds into the model

In `MtDNAEmbeddings`, the heteroplasmy scalar is projected into the embedding space:

```python
het_proj = self.het_norm(self.het_projection(het_values.unsqueeze(-1)))
emb = kmer_emb + circular_pe + het_proj
```

The projection is a `Linear(1, hidden_size)` layer followed by `LayerNorm`. This learned transformation allows the model to modulate the k-mer representation based on how heteroplasmically variable the position is.

**Why a linear projection instead of discretization?** Discretizing (e.g., high/medium/low) introduces an arbitrary threshold and discards information. The continuous projection is learned end-to-end and preserves the full signal.

---

## Vocabulary Statistics

For the human mtDNA corpus (HmtDB, 34,975 sequences used):

| Statistic | Value |
|---|---|
| Total unique k-mers observed | 4,068 of 4,096 (99.3%) |
| K-mers never observed | ~28 (all contain unusual base combinations) |
| Mean tokens per genome | 16,564 (≈ genome_length − k + 1) |
| Most frequent k-mer | varies by GC content; poly-C tracts (D-loop) dominate |
| `[UNK]` token rate | < 0.1% of tokens (N bases are rare in curated HmtDB) |

The near-complete coverage of the k-mer vocabulary means the model is unlikely to encounter out-of-vocabulary tokens even on divergent sequences (Neanderthal, Denisovan) not seen during training.
