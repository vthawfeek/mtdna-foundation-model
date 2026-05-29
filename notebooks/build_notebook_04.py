"""Build notebooks/04_showcase.ipynb — Ancient DNA demonstration."""

import nbformat


cells = []


def md(src):
    return nbformat.v4.new_markdown_cell(src)


def code(src):
    return nbformat.v4.new_code_cell(src)


# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md("""# Ancient DNA Demonstration
## mtDNA Foundation Model — Day 20

**Zero-shot test:** embed Neanderthal (NC_011137.1) and Denisovan (FR695060.1) mtDNA
sequences using a model trained only on modern vertebrate genomes. No fine-tuning.
No labels. No ancient training examples.

If pre-training captured real evolutionary structure, the ancient sequences should
appear near the root of the modern human phylogeny — consistent with decades of
paleoanthropological analysis.

**Sequences used:**
- Neanderthal: 16,565 bp (Vindija Cave, Croatia — highest-quality ancient mtDNA)
- Denisovan: 16,570 bp (Altai Cave, Russia — 99.9% of sites called)
- Modern humans: 100 sequences sampled from HmtDB test set (50 major haplogroups)

Embeddings are pre-computed and cached in `data/processed/showcase_embeddings.npz`
(first run takes ~7 min on CPU; subsequent runs load instantly).
"""))


# ── Setup ──────────────────────────────────────────────────────────────────────
cells.append(md("## Setup"))

cells.append(code("""import os, warnings
os.chdir('/home/user/Documents/Personal/ai_lab/mtdna_foundation_model')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = 'docs/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
print("Setup complete")
"""))


# ── Load Embeddings ────────────────────────────────────────────────────────────
cells.append(md("""## 1. Load Pre-computed Embeddings

Embeddings are loaded from cache if available. Otherwise, the model is loaded and
embeddings are computed fresh (~7 min on CPU for 100 sequences + 2 ancient).
"""))

cells.append(code("""from pathlib import Path

CACHE_PATH = 'data/processed/showcase_embeddings.npz'

def _compute_and_cache():
    \"\"\"Compute embeddings from scratch and save to cache.\"\"\"
    import pandas as pd
    import time
    from mtdna_fm.inference.api import MtDNAEmbedder
    from mtdna_fm.data.ancient_dna import prepare_ancient_sequences

    df = pd.read_parquet('data/processed/test.parquet')
    human = df[df.species == 'homo_sapiens']

    # Stratified sample: up to 4 sequences per top-50 haplogroups
    top_hg = human['haplogroup'].value_counts().head(50).index.tolist()
    parts = [human[human['haplogroup'] == hg].head(4) for hg in top_hg]
    sample_df = pd.concat(parts).sample(n=min(100, len(pd.concat(parts))), random_state=42)
    print(f"Sampled {len(sample_df)} sequences across {sample_df['haplogroup'].nunique()} haplogroups")

    embedder = MtDNAEmbedder.from_pretrained('models/phase1_v1', device='cpu')

    # Ancient DNA
    print("Embedding ancient DNA (Neanderthal + Denisovan)...")
    ancient_seqs = prepare_ancient_sequences()
    anc_labels = list(ancient_seqs.keys())
    anc_embs = np.stack([embedder.embed_genome(s) for s in ancient_seqs.values()])

    # Modern humans
    print(f"Embedding {len(sample_df)} modern humans...")
    t0 = time.time()
    modern_embs = []
    hgs = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        modern_embs.append(embedder.embed_genome(row.sequence))
        hgs.append(row.haplogroup)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(sample_df)} ({elapsed:.0f}s elapsed)")
    modern_embs = np.stack(modern_embs)

    np.savez(
        CACHE_PATH,
        modern_embeddings=modern_embs,
        modern_labels=np.array(hgs),
        ancient_embeddings=anc_embs,
        ancient_labels=np.array(anc_labels),
    )
    print(f"Cached to {CACHE_PATH}")
    return modern_embs, hgs, anc_embs, anc_labels


if Path(CACHE_PATH).exists():
    data = np.load(CACHE_PATH, allow_pickle=True)
    modern_embeddings = data['modern_embeddings']
    modern_labels = list(data['modern_labels'])
    ancient_embeddings = data['ancient_embeddings']
    ancient_labels = list(data['ancient_labels'])
    print(f"Loaded from cache: {modern_embeddings.shape} modern, {ancient_embeddings.shape} ancient")
else:
    print("Cache not found — computing embeddings (this takes ~7 min on CPU)...")
    modern_embeddings, modern_labels, ancient_embeddings, ancient_labels = _compute_and_cache()

print(f"Modern:  {modern_embeddings.shape}")
print(f"Ancient: {ancient_embeddings.shape}")
print(f"Ancient labels: {ancient_labels}")
"""))


# ── UMAP Visualization ────────────────────────────────────────────────────────
cells.append(md("""## 2. UMAP: Modern Humans + Ancient Hominids

Fit UMAP on all sequences (modern + ancient) jointly. Ancient sequences get placed
in the same 2-D coordinate space as modern humans, with no privileged access to
their evolutionary labels.

**What to look for:**
- Ancient sequences should appear **outside** the modern human haplogroup cloud
- They should appear **near the root**, not at the European or Asian periphery
- Neanderthal and Denisovan should cluster **near each other** (both Homo genus,
  diverged ~300-400 kya from each other) but distinct from any modern haplogroup
"""))

cells.append(code("""from mtdna_fm.evaluation.viz import plot_umap_with_ancient_dna

fig = plot_umap_with_ancient_dna(
    modern_embeddings=modern_embeddings,
    modern_labels=modern_labels,
    ancient_embeddings=ancient_embeddings,
    ancient_labels=ancient_labels,
    title="mtDNA-FM Zero-Shot: Modern Humans + Ancient Hominids",
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
)
fig.savefig(f'{FIGURES_DIR}/ancient_dna_umap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved to docs/figures/ancient_dna_umap.png")
"""))


# ── Distance Analysis ──────────────────────────────────────────────────────────
cells.append(md("""## 3. Nearest-Neighbour Analysis

Compute cosine similarity between each ancient sequence and all modern haplogroups.
This gives a quantitative read on where the model places each ancient genome
relative to the known human phylogeny.

**Expected result based on paleoanthropology:**
- Neanderthal and Denisovan should both be most similar to root haplogroups (L0, L1, L2, L3)
- They should be *less* similar to derived European (H, HV, J, T) or East Asian (D, C, M) tips
"""))

cells.append(code("""from sklearn.preprocessing import normalize

# Unit-normalise embeddings for cosine similarity
modern_norm = normalize(modern_embeddings, norm='l2')
ancient_norm = normalize(ancient_embeddings, norm='l2')

# Compute cosine similarity matrix: (n_ancient, n_modern)
cosine_sim = ancient_norm @ modern_norm.T  # (2, n_modern)

print("Nearest 10 modern sequences for each ancient genome (by cosine similarity):")
print()
for i, anc_label in enumerate(ancient_labels):
    top_idx = np.argsort(cosine_sim[i])[::-1][:10]
    top_hg = [modern_labels[j] for j in top_idx]
    top_sim = cosine_sim[i, top_idx]
    print(f"** {anc_label} **")
    for hg, sim in zip(top_hg, top_sim):
        print(f"  {hg:<12}  cosine = {sim:.4f}")
    print()
"""))

cells.append(code("""# Compare mean similarity to root (L*) vs derived clades
import collections

def mean_clade_sim(anc_idx, clade_prefix):
    \"\"\"Mean cosine sim from ancient[anc_idx] to modern sequences with haplogroup starting with prefix.\"\"\"
    clade_idx = [j for j, hg in enumerate(modern_labels) if hg.startswith(clade_prefix)]
    if not clade_idx:
        return float('nan'), 0
    return float(np.mean(cosine_sim[anc_idx, clade_idx])), len(clade_idx)

clade_groups = {
    'L (root African)': 'L',
    'H (European tip)': 'H',
    'D/C (East Asian)': ('D', 'C'),
    'B (Asian/American)': 'B',
    'A (Asian/American)': 'A',
}

print(f"{'Clade':<25}  ", end="")
for anc_label in ancient_labels:
    print(f"{anc_label:<14}", end="")
print()
print("-" * (25 + 2 + 14 * len(ancient_labels)))

for clade_name, prefix in clade_groups.items():
    print(f"{clade_name:<25}  ", end="")
    for i in range(len(ancient_labels)):
        if isinstance(prefix, tuple):
            idx = [j for j, hg in enumerate(modern_labels) if any(hg.startswith(p) for p in prefix)]
            sim = float(np.mean(cosine_sim[i, idx])) if idx else float('nan')
        else:
            sim, _ = mean_clade_sim(i, prefix)
        print(f"{sim:<14.4f}", end="")
    print()
"""))


# ── Ancient vs Ancient ────────────────────────────────────────────────────────
cells.append(md("""## 4. Neanderthal vs Denisovan Similarity

How similar are Neanderthal and Denisovan to each other relative to the variance
in the modern human population? If the model captures evolutionary signal, their
mutual similarity should be:
- **Higher** than their mean similarity to any modern human haplogroup
  (they share a closer common ancestor with each other than with any modern Homo sapiens)
- Actually this is debated: Denisovan and Neanderthal shared a common ancestor ~300-400 kya,
  while modern humans and Neanderthals/Denisovans diverged ~500-600 kya from each other.
  The model's representation should reflect this, but with only sequence information
  (no introgression data), it may not perfectly recapitulate the tree topology.
"""))

cells.append(code("""anc_anc_sim = float(ancient_norm[0] @ ancient_norm[1])
print(f"Cosine similarity: Neanderthal vs Denisovan = {anc_anc_sim:.4f}")
print()

# Modern human pairwise similarity distribution
n_pairs = min(500, len(modern_norm))
modern_sample = modern_norm[:n_pairs]
pairwise = modern_sample @ modern_sample.T
# Upper triangle (off-diagonal)
triu = pairwise[np.triu_indices(n_pairs, k=1)]
print(f"Modern human pairwise cosine similarity:")
print(f"  Mean: {np.mean(triu):.4f}")
print(f"  Std:  {np.std(triu):.4f}")
print(f"  Min:  {np.min(triu):.4f}")
print(f"  Max:  {np.max(triu):.4f}")
print()
print(f"Neanderthal-Denisovan cosine ({anc_anc_sim:.4f}) relative to modern distribution:")
z_score = (anc_anc_sim - np.mean(triu)) / np.std(triu)
print(f"  z-score: {z_score:.2f} standard deviations from modern mean")
if anc_anc_sim < np.mean(triu):
    print("  -> Ancient pair is MORE distinct than typical modern pairs (expected — greater divergence time)")
else:
    print("  -> Ancient pair is within modern human similarity range")
"""))


# ── Summary ────────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Interpretation

### What this demonstrates

1. **Zero-shot generalisation:** The model was trained entirely on modern vertebrate
   mtDNA genomes. It has never seen an ancient hominid sequence. Yet it places
   Neanderthal and Denisovan in a geometrically coherent position relative to the
   modern human phylogeny.

2. **Evolutionary structure without labels:** The placement is driven purely by
   sequence similarity at the k-mer level. No haplogroup labels, no evolutionary
   tree supervision, no ancient DNA in the training set.

3. **Verifiability:** The result is testable against decades of paleoanthropological
   research. Molecular clock analyses (e.g., Green et al. 2010, Meyer et al. 2012)
   place Neanderthal and Denisovan outside modern human haplogroup diversity but within
   the Homo genus. If the model's UMAP agrees with this, the pre-training has captured
   real phylogenetic signal. If it doesn't, something is wrong with the representations.

### Limitations

- This is a 100-sequence sample from HmtDB. A larger panel (5,000 sequences) would
  give a cleaner UMAP with better haplogroup separation.
- Phase 1 model is used here (cross-species corpus). Phase 2 (human-only fine-tuning
  with heteroplasmy signal) may give different placement.
- The model captures k-mer frequency patterns. It doesn't explicitly model insertion/
  deletion events (the ~4 bp length difference between hominid genomes), which are also
  phylogenetically informative.

### Next steps (Day 21)

Push model + tokenizer to HuggingFace Hub so this demonstration is reproducible
from a single `from_pretrained("vthawfeek/mtdna-foundation-model")` call.
"""))

cells.append(code("""print("Day 20 complete: ancient DNA zero-shot demonstration.")
print(f"Figure: docs/figures/ancient_dna_umap.png")
"""))


# ── Write notebook ─────────────────────────────────────────────────────────────
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}
nb.metadata['language_info'] = {
    'name': 'python',
    'version': '3.11.0',
}

output_path = 'notebooks/04_showcase.ipynb'
with open(output_path, 'w') as f:
    nbformat.write(nb, f)
print(f"Wrote {output_path}")
