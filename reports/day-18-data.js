(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[18] = {
    topic: "Heteroplasmy Regression",
    commit: "",
    status: "complete",
    built: [
      "MtDNAForHeteroplasmyRegression: regression head Linear(256,64)->GELU->Linear(64,1)->Sigmoid on variant-token hidden state, Huber loss",
      "HeteroplasmyRegressionOutput dataclass; exported from model/__init__.py",
      "HeteroplasmyRegressionDataset: windowed dataset with mean het_level target, synthetic fallback for testing",
      "finetune_heteroplasmy(): 5-fold cross-validation, R-squared + Spearman rho per fold, final model saved on all data",
      "CLI dispatch wired: 'heteroplasmy' task now calls finetune_heteroplasmy()",
      "configs/finetuning_heteroplasmy.yaml: LoRA r=4, n_folds=5, huber_delta=0.1, batch=16, 15 epochs",
      "12 model tests (shape, range, loss, gradients, freeze/unfreeze, LoRA, architecture)",
      "8 dataset + CLI tests (synthetic fallback, parquet loading, window size, label range, dispatch)"
    ],
    learned: [
      "Huber vs MSE: Huber down-weights large residuals, robust to noisy gnomAD heteroplasmy estimates from small carrier populations",
      "5-fold CV vs held-out split: with ~1,000 points a fixed split wastes training signal; CV uses all data and reports variance across folds",
      "Spearman rho vs R-squared: ranking accuracy matters more than absolute accuracy for biological constraint studies",
      "Variant-token hidden state: heteroplasmy level is a local property of nucleotide context, not a global genome property — same inductive bias as pathogenicity",
      "LoRA r=4 for small datasets: heavier regularisation than r=8 (haplogroup) prevents overfitting on ~1,000 training examples"
    ],
    decisions: [
      "Huber delta=0.1: residuals in [0,1] space rarely exceed 0.5; delta=0.1 puts the linear/squared transition at biologically meaningful scale",
      "Sigmoid output: bounds predictions to [0,1] without clipping at inference time",
      "5-fold CV: dataset too small for a held-out split; cross-validation maximises both training signal and evaluation reliability",
      "Synthetic fallback in HeteroplasmyRegressionDataset: matches PathogenicityVariantDataset pattern for test coverage without shipping real data",
      "Spearman > 0.30 as real-signal threshold: modest but meaningful bar for a regression on noisy gnomAD estimates"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <!-- Scatter plot with regression line -->
          <rect x="10" y="10" width="100" height="100" rx="4" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
          <!-- Axes -->
          <line x1="20" y1="100" x2="105" y2="100" stroke="#495057" stroke-width="1.5"/>
          <line x1="20" y1="100" x2="20" y2="15" stroke="#495057" stroke-width="1.5"/>
          <!-- Axis labels -->
          <text x="62" y="115" text-anchor="middle" font-size="8" fill="#868e96">true het level</text>
          <text x="7" y="60" text-anchor="middle" font-size="8" fill="#868e96" transform="rotate(-90,7,60)">predicted</text>
          <!-- Scatter dots -->
          <circle cx="32" cy="88" r="3" fill="#339af0" opacity="0.8"/>
          <circle cx="42" cy="79" r="3" fill="#339af0" opacity="0.8"/>
          <circle cx="50" cy="72" r="3" fill="#339af0" opacity="0.8"/>
          <circle cx="58" cy="65" r="3" fill="#339af0" opacity="0.8"/>
          <circle cx="65" cy="57" r="3" fill="#339af0" opacity="0.8"/>
          <circle cx="75" cy="48" r="3" fill="#339af0" opacity="0.8"/>
          <circle cx="83" cy="40" r="3" fill="#339af0" opacity="0.8"/>
          <circle cx="93" cy="30" r="3" fill="#339af0" opacity="0.8"/>
          <!-- Outlier -->
          <circle cx="55" cy="40" r="3" fill="#fa5252" opacity="0.8"/>
          <!-- Regression line -->
          <line x1="25" y1="95" x2="100" y2="22" stroke="#2f9e44" stroke-width="2"/>
        </svg>
        <div class="eli5-caption">Predicting heteroplasmy level</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>Heteroplasmy is the fraction of your mitochondria that carry a mutation. Some people have 10% mutant copies, others 80%. This fraction (0 to 1) is what we're trying to predict from the DNA sequence alone.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>The model reads a 512-letter window of DNA around a mutation and outputs a single number between 0 and 1. Higher values mean the mutation tends to persist at high levels in cells.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>We train on ~1,000 real mutations from gnomAD (a human variant database) where the average heteroplasmy level was measured across many people. We use Huber loss — a training signal that ignores extreme outliers so they don't skew the whole model.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>We evaluate with 5-fold cross-validation: split the data into 5 chunks, train on 4, test on 1, repeat. If Spearman correlation &gt; 0.30, the model is ranking variants by constraint better than random — a real biological signal.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">Huber Loss</div>
      <div class="math-eq">L(y, yhat) = { 0.5*(y-yhat)^2          if |y-yhat| &lt;= delta
             { delta*(|y-yhat| - 0.5*delta)  otherwise

delta = 0.1 (transition from squared to linear at residual = 0.1)</div>
      <div class="math-example"><strong>Example:</strong> true het_level = 0.4, predicted = 0.55, residual = 0.15 &gt; delta=0.1
Huber: 0.1*(0.15 - 0.05) = 0.010  vs  MSE: 0.5*(0.15)^2 = 0.011
At residual = 0.5 (outlier): Huber: 0.1*(0.5-0.05) = 0.045  vs  MSE: 0.5*(0.5)^2 = 0.125
Huber reduces the outlier's contribution by 65% compared to MSE.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Spearman Rank Correlation</div>
      <div class="math-eq">rho = 1 - (6 * sum(d_i^2)) / (n*(n^2 - 1))
where d_i = rank(y_i) - rank(yhat_i)</div>
      <div class="math-example"><strong>Example:</strong> 4 variants with true levels [0.1, 0.3, 0.6, 0.8], predicted [0.15, 0.25, 0.55, 0.9]
True ranks: [1,2,3,4]  Predicted ranks: [1,2,3,4]  d_i = [0,0,0,0]
rho = 1 - 0 = 1.0  (perfect ranking, even if absolute values differ)
Contrast with R^2 which penalises the 0.25 vs 0.3 difference.</div>
    </div>`
  };
})();
