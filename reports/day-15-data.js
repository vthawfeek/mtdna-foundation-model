(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[15] = {
    topic: "Genome Embedding API",
    commit: "56666c5",
    status: "complete",
    built: [
      "mtdna_fm/inference/api.py — MtDNAEmbedder: embed_genome (CLS-mean pooling), embed_variant (token-level), embed_dataset, from_pretrained",
      "tests/test_inference.py — 18 tests covering all public methods and from_pretrained loading"
    ],
    learned: [
      "CLS-mean pooling: mean the CLS token across all overlapping windows for a full-genome embedding that weights every region equally",
      "from_pretrained loads MtDNAForMaskedModeling then extracts .mtdna — discards prediction heads, mirrors HuggingFace BertModel pattern",
      "Token-level embedding for variants: hidden state at the variant's token position captures local functional context (codon, tRNA fold, regulatory motif)",
      "k = log4(vocab_size − 6) infers k-mer size from vocabulary without storing it separately — keeps the API clean"
    ],
    decisions: [
      "Extract .mtdna from pretraining wrapper rather than stripping weight prefixes — cleaner, validates checkpoint format",
      "embed_dataset delegates to embed_genome per-sequence: avoids cross-sequence batching complexity for marginal CPU gain",
      "Circular window wrapping at token stream boundary — matches pre-training topology, ensures 16568/0 junction is covered",
      "Phase 1 embeddings used for verification: zero-shot 3-NN 50% vs 12.5% random (8 haplogroups), exceeds ≥40% target"
    ],
    eli5: null,
    math: null
  };
})();
