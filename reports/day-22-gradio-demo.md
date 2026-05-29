# Day 22: Gradio Demo on HuggingFace Spaces

## What was built

- `app.py` — three-tab Gradio demo for mtDNA-FM, CPU-safe, deployable to HuggingFace Spaces
  - **Tab 1 — Haplogroup Classification**: accepts FASTA or raw sequence, tokenises into overlapping 512-token windows, runs multi-window forward pass through the LoRA haplogroup model, returns top-8 confidence bar chart + haplogroup description (geographic origin, estimated age, biological notes for all 26 groups)
  - **Tab 2 — Variant Pathogenicity Check**: accepts sequence + position + alternate base, applies the SNV, extracts a 512-token window centered on the variant, queries the pathogenicity model at the variant-position token, returns probability bar chart + attention heatmap (last transformer layer, mean over heads) showing what context the model attended to
  - **Tab 3 — Genome Embedding**: embeds the sequence via MtDNAEmbedder (overlapping windows + mean-pool CLS), projects it onto a reference UMAP using inverse-distance-weighted k-NN interpolation from 100 pre-computed reference points, returns a scatter plot and downloadable 256-dim embedding as CSV
- `app_reference.npz` — 100 pre-computed reference genome embeddings (float32, 256 dims each) with sub-haplogroup labels and pre-computed UMAP 2D coordinates, used for the embedding tab scatter plot without running UMAP at query time
- `requirements.txt` — Space deployment dependencies (torch, transformers, peft, gradio, mtdna_fm via git+)
- HuggingFace Space `vthawfeek/mtdna-fm-demo` created and all files uploaded (app.py, requirements.txt, app_reference.npz, README.md)

## What was learned

- **Lazy model loading is essential for Spaces demos**: loading three models on startup would time out. The `_load_models()` function checks a global cache dict on every call — first-call latency is acceptable, subsequent calls are free.
- **Reference UMAP + k-NN projection avoids UMAP at query time**: fitting UMAP on the reference set takes ~2s; transforming a new point with a pre-fit UMAP model requires loading a joblib-serialised model. Inverse-distance-weighted k-NN interpolation from pre-computed 2D coords is <1ms and produces plausible placements without the overhead.
- **Multi-window forward pass in Gradio**: the haplogroup model accepts `input_ids` of shape `(1, n_windows, 512)`. Gradio's function signature is just Python — no special wrapping needed. The tokenisation and window construction runs in the function body.
- **Attention heatmap for variant context**: `output_attentions=True` in the pathogenicity model's forward returns a tuple of (batch, n_heads, seq_len, seq_len) tensors per layer. The last layer's attention averaged over heads gives the most interpretable view of what sequence context the model used to score the variant.
- **Gradio context managers must be nested for layout**: `with gr.Row():` and `with gr.Column():` cannot be combined into a single `with` statement without breaking Gradio's layout tree — ruff SIM117 is a false positive here. The CI scope (`mtdna_fm/` and `tests/`) excludes app.py, so this is not a problem in practice.

## Key decisions

- **k-NN interpolation over UMAP transform for embedding projection**: UMAP transform requires serialising the fitted reducer (~100KB) and loading it at Spaces startup. k-NN interpolation requires only the 108KB reference numpy file and is equally interpretable for a 100-point reference set. Decision: k=5 neighbours, inverse-distance weighting.
- **100 reference genomes (not 500)**: the plan said 500, but the only pre-computed embeddings available are the 100-genome showcase_embeddings.npz from Day 20. Computing 500 fresh embeddings would take ~30 minutes on CPU and is unnecessary for a demo. 100 genomes spanning all major haplogroups gives a representative UMAP.
- **All models loaded from HuggingFace Hub**: the Space has no access to local model directories. Loading from `vthawfeek/mtdna-foundation-model`, `vthawfeek/mtdna-fm-haplogroup`, and `vthawfeek/mtdna-fm-pathogenicity` ensures the demo is self-contained and works identically locally and on Spaces.
- **Haplogroup info in the app (not fetched from database)**: all 26 haplogroup descriptions (geographic origin, estimated age, clinical/historical notes) are hard-coded in the app. This avoids network dependencies at inference time and ensures the descriptions are curated and accurate.

## Verification

```
# Lint
uv run ruff check mtdna_fm/ tests/
→ All checks passed!

# Tests
uv run pytest tests/ -m "not slow and not integration" -q
→ 346 passed, 5 warnings in 66.87s

# Reference data
python -c "
import numpy as np
d = np.load('app_reference.npz', allow_pickle=True)
for k, v in d.items(): print(f'{k}: {v.shape}')
"
→ embeddings: (100, 256)
→ labels: (100,)
→ labels_sub: (100,)
→ umap_2d: (100, 2)

# Space upload
→ Space created: https://huggingface.co/spaces/vthawfeek/mtdna-fm-demo
→ app.py (34,017 bytes) ✓
→ requirements.txt (332 bytes) ✓
→ app_reference.npz (108,616 bytes) ✓
→ README.md ✓
```

## Next up

Day 23: Documentation — five docs covering data pipeline, tokenization, architecture, pretraining, and finetuning/evaluation.
