(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[19] = {
    topic: "Evaluation Framework",
    commit: "",
    status: "complete",
    built: [
      "mtdna_fm/evaluation/haplogroup_eval.py: accuracy, macro-F1, per-haplogroup breakdown, confusion matrix",
      "mtdna_fm/evaluation/variant_eval.py: AUROC, AUPRC, per-variant-type breakdown (missense/tRNA/rRNA/D-loop/other)",
      "mtdna_fm/evaluation/viz.py: UMAP, ROC curve, confusion matrix heatmap, attention weight heatmap",
      "mtdna_fm/evaluation/__init__.py: clean public surface for all evaluation utilities",
      "mtdna_fm/scripts/evaluate.py: mtdna-evaluate CLI, writes eval_summary.json, --synthetic smoke-test flag",
      "notebooks/03_finetuning_results.ipynb: UMAP, confusion matrix, ROC curve, attention heatmap",
      "tests/test_evaluation.py: 33 new tests covering all evaluation modules",
      "tests/test_scripts.py: updated evaluate CLI tests (replaced stale stubs)"
    ],
    learned: [
      "NumPy 2.x removed np.trapz — must use np.trapezoid; silent breakage when upgrading environments",
      "AUROC computed without sklearn by building the ROC curve from scratch and applying the trapezoidal rule",
      "AUPRC boundary handling: anchor precision=1 at recall=0 to avoid underestimating the area",
      "Per-variant-type breakdowns (tRNA vs missense vs D-loop) give more diagnostic signal than a single scalar",
      "UMAP of genome embeddings is the single most important diagnostic — phylogenetic topology emerging from sequence alone validates pre-training",
      "Hand-coded metrics are more auditable than sklearn black-boxes for a foundation model others will use"
    ],
    decisions: [
      "No sklearn dependency: evaluation code must be auditable and dependency-minimal; hand-coded metrics are inspectable",
      "--synthetic flag: smoke-tests the full evaluate pipeline in CI without a trained checkpoint",
      "Variant-type by coordinate range (not VEP annotation): avoids annotation dependency at eval time; rCRS ranges are stable",
      "Downsample ROC/PR curves to ≤200 points before JSON: prevents eval_summary.json from becoming unreadably large"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <!-- UMAP scatter plot illustration -->
          <!-- L clade cluster (red) -->
          <circle cx="30" cy="90" r="5" fill="#DC143C" opacity="0.8"/>
          <circle cx="38" cy="85" r="4" fill="#A52A2A" opacity="0.8"/>
          <circle cx="25" cy="95" r="4" fill="#CD5C5C" opacity="0.8"/>
          <!-- M clade cluster (green) -->
          <circle cx="85" cy="80" r="5" fill="#228B22" opacity="0.8"/>
          <circle cx="78" cy="75" r="4" fill="#32CD32" opacity="0.8"/>
          <circle cx="90" cy="88" r="4" fill="#006400" opacity="0.8"/>
          <!-- H clade cluster (orange) -->
          <circle cx="75" cy="35" r="5" fill="#FFA500" opacity="0.8"/>
          <circle cx="82" cy="28" r="4" fill="#FF8C00" opacity="0.8"/>
          <circle cx="70" cy="30" r="4" fill="#FFD700" opacity="0.8"/>
          <!-- N/R cluster (blue) -->
          <circle cx="55" cy="50" r="5" fill="#4169E1" opacity="0.8"/>
          <circle cx="62" cy="45" r="4" fill="#00008B" opacity="0.8"/>
          <circle cx="50" cy="55" r="4" fill="#6495ED" opacity="0.8"/>
          <!-- Arrow from L to N (Out of Africa) -->
          <line x1="35" y1="88" x2="50" y2="58" stroke="#666" stroke-width="1.5" marker-end="url(#arr)" opacity="0.6"/>
          <!-- Arrow from N to H (European) -->
          <line x1="57" y1="48" x2="70" y2="38" stroke="#666" stroke-width="1.5" marker-end="url(#arr)" opacity="0.6"/>
          <defs>
            <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#666"/>
            </marker>
          </defs>
          <!-- Axis labels -->
          <text x="5" y="128" font-size="8" fill="#888">UMAP 1</text>
          <text x="2" y="80" font-size="8" fill="#888" transform="rotate(-90,8,80)">UMAP 2</text>
        </svg>
        <div class="eli5-caption">Genome embeddings form the phylogenetic tree</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>After training, we need to check: did the model actually learn anything useful? We do this by running the model on data it has never seen before — a held-out test set — and measuring how often it gets the right answer.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>For haplogroup classification, we count how many genomes are correctly labelled and compute a number called "macro-F1" that penalises the model equally for every haplogroup, even rare ones.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>For variant pathogenicity, we use "AUROC" — think of it as: if we randomly pick one pathogenic and one benign variant, how often does the model score the pathogenic one higher? A score of 1.0 is perfect; 0.5 is no better than a coin flip.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>The most important check is the UMAP plot: we compress 256-dimensional genome embeddings down to 2 dimensions and look at whether related haplogroups cluster together. If African L0/L1/L2 sit at the base, L3 branches off, and European H/HV appears at the tip — that's the phylogenetic tree emerging from sequence alone, with no labels used during training.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">AUROC — Area Under the ROC Curve</div>
      <div class="math-eq">
        At each threshold t:
          FPR(t) = FP(t) / (FP(t) + TN(t))
          TPR(t) = TP(t) / (TP(t) + FN(t))

        AUROC = integral of TPR d(FPR) = sum over thresholds of
                (FPR[i+1] - FPR[i]) * (TPR[i+1] + TPR[i]) / 2
                (trapezoidal rule)
      </div>
      <div class="math-example"><strong>Example:</strong> 6 variants, scores [0.1, 0.2, 0.3, 0.7, 0.8, 0.9], labels [0,0,0,1,1,1].
At threshold 0.65: TP=3, FP=0, FN=0, TN=3 → TPR=1.0, FPR=0.0.
At threshold 0.25: TP=3, FP=2, FN=0, TN=1 → TPR=1.0, FPR=0.67.
Trapezoidal area → AUROC = 1.0 (perfect separation).</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Macro-F1 for Haplogroup Classification</div>
      <div class="math-eq">
        Per-class F1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
        where precision_c = TP_c / (TP_c + FP_c)
              recall_c    = TP_c / (TP_c + FN_c)

        Macro-F1 = mean( F1_c ) over all classes c with support > 0
      </div>
      <div class="math-example"><strong>Example:</strong> 3 haplogroups, F1 scores [0.95, 0.90, 0.88].
Macro-F1 = (0.95 + 0.90 + 0.88) / 3 = 0.91.
Each class contributes equally regardless of how many samples it has.</div>
    </div>`
  };
})();
