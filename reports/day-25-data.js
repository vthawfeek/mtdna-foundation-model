(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[25] = {
    topic: "Showcase Notebook",
    commit: "10ecc7b",
    status: "complete",
    built: [
      "notebooks/04_showcase.ipynb — 7-section self-contained showcase notebook",
      "notebooks/build_notebook_04.py — updated build script",
      "docs/figures/showcase_tsne.png — t-SNE of 100 human genome embeddings by haplogroup",
      "docs/figures/showcase_confusion_matrix.png — 26x26 haplogroup confusion matrix",
      "docs/figures/showcase_roc_curve.png — variant pathogenicity ROC curve (AUROC = 0.877)",
      "docs/figures/showcase_ancient_dna_umap.png — UMAP with ancient hominids overlaid",
      "docs/figures/showcase_gene_type_recovery.png — t-SNE of 37 mtDNA gene embeddings"
    ],
    learned: [
      "Embedding space reflects phylogeny without labels: t-SNE clusters show haplogroup structure from pre-training alone",
      "Gene-type recovery from sequence: protein/tRNA/rRNA genes cluster by function without any gene-type supervision",
      "Confusion errors are phylogenetically informative: classifier mistakes concentrate within clades, not across distant branches",
      "Silhouette score quantifies embedding quality and is more reliable than visual t-SNE inspection",
      "Ancient DNA placement is the hardest zero-shot test: consistent with paleoanthropological consensus with no fine-tuning"
    ],
    decisions: [
      "Use cached embeddings (100 sequences) rather than fresh 5,000: notebook runs under 5 minutes on CPU while showing the same methodological content",
      "embed_variant for gene-type recovery (not embed_genome): position-specific hidden states capture per-gene context better than mean-pooled genome vectors",
      "Silhouette score over visual inspection: quantitative metric on full 256-d space is more reproducible than 2-D t-SNE projection",
      "Rebuild Day 20 notebook completely: Day 20 was an ancient DNA stub; Day 25 needs the full project narrative"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <!-- Phylogenetic tree emerging from embeddings -->
          <circle cx="60" cy="110" r="6" fill="#DC143C" opacity="0.9"/>
          <line x1="60" y1="104" x2="30" y2="75" stroke="#888" stroke-width="1.5"/>
          <line x1="60" y1="104" x2="90" y2="75" stroke="#888" stroke-width="1.5"/>
          <circle cx="30" cy="72" r="5" fill="#8B0000" opacity="0.9"/>
          <circle cx="90" cy="72" r="5" fill="#006400" opacity="0.9"/>
          <line x1="30" y1="67" x2="15" y2="45" stroke="#888" stroke-width="1.5"/>
          <line x1="30" y1="67" x2="45" y2="45" stroke="#888" stroke-width="1.5"/>
          <line x1="90" y1="67" x2="75" y2="45" stroke="#888" stroke-width="1.5"/>
          <line x1="90" y1="67" x2="105" y2="45" stroke="#888" stroke-width="1.5"/>
          <circle cx="15" cy="42" r="4" fill="#A52A2A"/>
          <circle cx="45" cy="42" r="4" fill="#CD5C5C"/>
          <circle cx="75" cy="42" r="4" fill="#228B22"/>
          <circle cx="105" cy="42" r="4" fill="#32CD32"/>
          <!-- Stars for ancient DNA -->
          <text x="10" y="20" font-size="14" fill="#FFD700">★</text>
          <text x="95" y="20" font-size="14" fill="#9932CC">★</text>
          <line x1="18" y1="22" x2="30" y2="35" stroke="#FFD700" stroke-width="1" stroke-dasharray="3,2"/>
          <line x1="102" y1="22" x2="90" y2="35" stroke="#9932CC" stroke-width="1" stroke-dasharray="3,2"/>
          <text x="3" y="15" font-size="7" fill="#FFD700">Nean</text>
          <text x="88" y="15" font-size="7" fill="#9932CC">Deni</text>
        </svg>
        <div class="eli5-caption">Phylogeny from embeddings</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>The model reads each full mitochondrial genome (16,569 letters of DNA) and squishes it into a list of 256 numbers — its "fingerprint". Similar genomes get similar fingerprints.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>When we plot 100 human fingerprints in 2-D, related family groups (haplogroups) cluster together — just like a family tree. The model learned this from sequences alone, with no family-tree labels during training.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>We then fed in Neanderthal and Denisovan genomes the model had never seen. It placed them outside modern human diversity, near the root of the tree — exactly where paleoanthropologists say they belong based on decades of fossil and genetic evidence.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>Finally, we embedded each of the 37 mitochondrial genes individually. The model separated protein-coding genes, tRNA genes, and rRNA genes into distinct clusters — without ever being told which genes were which type. It recovered functional categories from sequence structure alone.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">t-SNE projection for visualisation</div>
      <div class="math-eq">minimize KL(P || Q) where P_ij = p(j|i) in 256-d space, Q_ij = p(j|i) in 2-d space
p(j|i) = exp(-||x_i - x_j||^2 / 2σ_i^2) / sum_k exp(-||x_i - x_k||^2 / 2σ_i^2)</div>
      <div class="math-example"><strong>Example:</strong> Two haplogroup-H genomes have embedding distance 0.12; one H and one L0 have distance 0.41. t-SNE perplexity=15 → H sequences compress into one cluster, L0 in a separate cluster. The KL divergence loss forces the 2-D map to preserve these distance ratios.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Silhouette score for embedding quality</div>
      <div class="math-eq">s(i) = (b(i) - a(i)) / max(a(i), b(i))
a(i) = mean distance to points in same cluster
b(i) = mean distance to points in nearest other cluster
S = mean over all points i</div>
      <div class="math-example"><strong>Example:</strong> Haplogroup H embeddings: a(i)=0.08 (tight within H), b(i)=0.35 (far from L-clade). s(i) = (0.35-0.08)/0.35 = 0.77. Mean S &gt; 0 confirms clades are geometrically separated in 256-d space.</div>
    </div>`
  };
})();
