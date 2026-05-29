(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[16] = {
    topic: "Haplogroup Fine-tuning",
    commit: "",
    status: "complete",
    built: [
      "MtDNAForHaplogroupClassification(PreTrainedModel): encoder + Linear(256, 26) head, supports single-window and multi-window inputs with CLS mean-pooling",
      "HaplogroupClassificationOutput: typed output dataclass (loss, logits, hidden_states, attentions)",
      "configs/finetuning_haplogroup.yaml: LoRA r=8, lora_alpha=16, target_modules=[query,key,value,dense], lr=1e-3, 20 epochs",
      "mtdna_fm/scripts/finetune.py: full CLI — loads Phase 2 checkpoint, applies PEFT LoRA, trains HaplogroupWindowDataset, saves best checkpoint and eval_metrics.json",
      "TestMtDNAForHaplogroupClassification: 10 tests — shapes, loss, gradients, freeze/unfreeze, LoRA, convergence",
      "mtdna_fm/model/__init__.py: exported new class and output type"
    ],
    learned: [
      "LoRA rank r=8 is appropriate for 47k labelled sequences; r=4 suits smaller datasets — rank is a regularisation knob, not just a capacity setting",
      "Mean-pooling CLS tokens across overlapping windows gives a whole-genome embedding where every region contributes equally — same strategy as sentence-transformers for long documents",
      "PEFT get_peft_model() wraps any PreTrainedModel by layer name — target_modules=['query','key','value','dense'] injects LoRA into all matching Linear layers across every transformer layer simultaneously",
      "Multi-window training via 3D input_ids (batch, n_windows, seq_len) reshaped to (batch*n_windows, seq_len) lets PyTorch process all windows in one forward pass with no Python loop",
      "freeze_encoder/unfreeze_encoder gives runtime control over the head-only vs full fine-tuning tradeoff without changing model architecture"
    ],
    decisions: [
      "Model accepts 2D and 3D input_ids: 2D for standard training, 3D for whole-genome inference in one call — avoids forcing callers to flatten manually",
      "26 major PhyloTree haplogroups as fixed label set: sub-haplogroups folded to major branch — biologically meaningful granularity for a first classification model",
      "HaplogroupWindowDataset repeats label across all windows of a genome — more gradient signal per genome, standard practice for long-sequence classification",
      "eval_metrics.json saved alongside checkpoint in DVC-compatible format — makes dvc metrics show work without extra code"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <!-- Base model (frozen) -->
          <rect x="10" y="10" width="100" height="40" rx="6" fill="#3b82f6" opacity="0.85"/>
          <text x="60" y="35" text-anchor="middle" font-size="11" fill="white" font-weight="bold">Pretrained Encoder</text>
          <!-- LoRA injection arrow -->
          <text x="60" y="62" text-anchor="middle" font-size="9" fill="#6366f1">+ LoRA adapters</text>
          <!-- Arrow down -->
          <line x1="60" y1="65" x2="60" y2="75" stroke="#6b7280" stroke-width="1.5" marker-end="url(#arr)"/>
          <defs>
            <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#6b7280"/>
            </marker>
          </defs>
          <!-- Classification head -->
          <rect x="25" y="76" width="70" height="28" rx="6" fill="#10b981" opacity="0.9"/>
          <text x="60" y="94" text-anchor="middle" font-size="10" fill="white" font-weight="bold">Classifier Head</text>
          <!-- Arrow down -->
          <line x1="60" y1="104" x2="60" y2="113" stroke="#6b7280" stroke-width="1.5" marker-end="url(#arr)"/>
          <!-- Output label -->
          <rect x="30" y="114" width="60" height="14" rx="4" fill="#f59e0b" opacity="0.9"/>
          <text x="60" y="124" text-anchor="middle" font-size="9" fill="white">Haplogroup H</text>
        </svg>
        <div class="eli5-caption">LoRA fine-tuning: tiny adapters steer a frozen giant</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>We have a big pretrained model that learned patterns in 30,000 genomes. We want to teach it one new job: "look at this DNA sequence and tell me which haplogroup family it belongs to." There are 26 families labelled A through X.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>Instead of retraining all 6 million parameters from scratch, LoRA adds tiny trainable "adapter" matrices (about 500KB total) alongside the frozen pretrained weights. Only those adapters update — the base model stays exactly as it was.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>A genome is 16,569 bases — too long to process in one go. We split it into ~63 overlapping 512-token windows, run the encoder on each, take the summary CLS token from each window, and average them into one genome-level vector.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>That genome vector goes through a small linear layer (256 → 26 numbers), we pick the highest number as the predicted haplogroup, and compute cross-entropy loss against the true label. Expected accuracy after training: above 95%.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">LoRA weight update</div>
      <div class="math-eq">W' = W + (B × A)  where  W frozen, A ∈ R^(r×d_in), B ∈ R^(d_out×r)
r = 8 (rank), d_in = d_out = 256 (hidden size)
trainable params per layer = 2 × 256 × 8 = 4,096  vs  256 × 256 = 65,536 full</div>
      <div class="math-example"><strong>Example:</strong> The query projection in layer 3 is a 256×256 matrix (65,536 params). LoRA replaces it with W + B×A where A is 8×256 and B is 256×8. Trainable count: 4,096 — 16× fewer. Across 6 layers × 4 modules = 24 LoRA pairs ≈ 98,304 adapter params vs 1,572,864 full fine-tune params for those layers.</div>
    </div>
    <div class="math-block">
      <div class="math-heading">Whole-genome embedding via CLS mean-pooling</div>
      <div class="math-eq">g = (1/N) × Σ_i CLS(window_i)   where N = ceil(L / stride)
L = 16,569 tokens, window = 512, stride = 256 → N ≈ 63 windows
g ∈ R^256 is the genome-level embedding used for classification</div>
      <div class="math-example"><strong>Example:</strong> Window 0 covers positions 0–511, window 1 covers 256–767, …, window 62 wraps around the circular junction. Each window's CLS token is a 256-dim vector. Averaging 63 such vectors gives the genome embedding g. Cross-entropy loss: CE(Linear(g), haplogroup_label).</div>
    </div>`
  };
})();
