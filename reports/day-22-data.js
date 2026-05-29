(function () {
  window.__dayData = window.__dayData || {};
  window.__dayData[22] = {
    topic: "Gradio Demo",
    commit: "c25a77f",
    status: "complete",
    built: [
      "app.py — three-tab Gradio demo: haplogroup classification, variant pathogenicity, genome embedding",
      "app_reference.npz — 100 pre-computed reference genome embeddings with UMAP 2D coords for the embedding tab scatter plot",
      "requirements.txt — HuggingFace Spaces deployment dependencies (torch, transformers, peft, gradio, mtdna_fm via git+)",
      "HuggingFace Space vthawfeek/mtdna-fm-demo — created and all files uploaded (app.py, requirements.txt, app_reference.npz, README.md)"
    ],
    learned: [
      "Lazy model loading is essential for Spaces: checking a global cache dict on every call avoids startup timeouts while keeping subsequent calls free",
      "Reference UMAP + k-NN projection avoids running UMAP at query time: inverse-distance-weighted k-NN from pre-computed 2D coords is <1ms and produces plausible embeddings",
      "Multi-window haplogroup inference in Gradio: input_ids shape (1, n_windows, 512) works natively in plain Python function bodies — no special wrapping",
      "Attention heatmap for variant context: last-layer attention averaged over heads gives the most interpretable view of what sequence context influenced pathogenicity scoring",
      "Gradio context managers must be nested for layout: combining with gr.Row() and with gr.Column() into one statement breaks Gradio's layout tree"
    ],
    decisions: [
      "k-NN interpolation over UMAP transform: avoids serialising/loading a UMAP model at Spaces startup; 108KB numpy file + k=5 neighbours is sufficient for a 100-point reference",
      "100 reference genomes not 500: only 100 pre-computed embeddings available from Day 20 showcase_embeddings.npz; recomputing 500 would take ~30min on CPU unnecessarily",
      "All models loaded from HuggingFace Hub: Space has no local model directories, Hub loading works identically locally and on Spaces",
      "Haplogroup descriptions hard-coded in app: avoids network dependencies at inference time; 26 curated entries covering geographic origin, age, and clinical/historical context"
    ],
    eli5: `<div class="eli5-wrap">
      <div class="eli5-art">
        <svg width="120" height="130" viewBox="0 0 120 130">
          <!-- DNA strand icon with browser window -->
          <!-- Browser window frame -->
          <rect x="5" y="5" width="110" height="90" rx="6" ry="6" fill="#e8f0fe" stroke="#4169E1" stroke-width="2"/>
          <rect x="5" y="5" width="110" height="18" rx="6" ry="6" fill="#4169E1"/>
          <rect x="5" y="14" width="110" height="9" fill="#4169E1"/>
          <!-- Browser dots -->
          <circle cx="16" cy="14" r="3" fill="#FF5F57"/>
          <circle cx="27" cy="14" r="3" fill="#FEBC2E"/>
          <circle cx="38" cy="14" r="3" fill="#28C840"/>
          <!-- Tab 1 active -->
          <rect x="50" y="27" width="20" height="8" rx="2" fill="#FFA500" opacity="0.9"/>
          <text x="60" y="34" text-anchor="middle" font-size="4" fill="white">🧬 H</text>
          <!-- Tab 2 -->
          <rect x="73" y="27" width="20" height="8" rx="2" fill="#6c757d" opacity="0.6"/>
          <text x="83" y="34" text-anchor="middle" font-size="4" fill="white">⚠️ V</text>
          <!-- Tab 3 -->
          <rect x="96" y="27" width="18" height="8" rx="2" fill="#6c757d" opacity="0.6"/>
          <text x="105" y="34" text-anchor="middle" font-size="4" fill="white">📊</text>
          <!-- Sequence input box -->
          <rect x="12" y="40" width="60" height="14" rx="2" fill="white" stroke="#adb5bd" stroke-width="1"/>
          <text x="15" y="51" font-size="5" fill="#6c757d">GATCACAGG...</text>
          <!-- Classify button -->
          <rect x="76" y="40" width="36" height="14" rx="3" fill="#4169E1"/>
          <text x="94" y="51" text-anchor="middle" font-size="5" fill="white">Classify</text>
          <!-- Bar chart output -->
          <rect x="12" y="61" width="38" height="5" rx="1" fill="#FFA500"/>
          <text x="53" y="66" font-size="4.5" fill="#333">H  42%</text>
          <rect x="12" y="68" width="22" height="5" rx="1" fill="#FF8C00"/>
          <text x="37" y="73" font-size="4.5" fill="#333">HV 24%</text>
          <rect x="12" y="75" width="14" height="5" rx="1" fill="#3CB371"/>
          <text x="29" y="80" font-size="4.5" fill="#333">U  15%</text>
          <!-- Title below -->
          <text x="60" y="108" text-anchor="middle" font-size="8" font-weight="bold" fill="#4169E1">Live Demo</text>
          <text x="60" y="120" text-anchor="middle" font-size="6" fill="#555">HuggingFace Spaces</text>
        </svg>
        <div class="eli5-caption">mtDNA-FM Gradio app</div>
      </div>
      <div class="eli5-text">
        <div class="eli5-step"><span class="eli5-num">1</span>We built a website where anyone can paste their mitochondrial DNA sequence and immediately get useful results — no programming required. It has three tabs, like pages in a notebook.</div>
        <div class="eli5-step"><span class="eli5-num">2</span>Tab 1 tells you which of the 26 major family groups (haplogroups) your DNA belongs to — like tracing whether your maternal ancestors came from Africa, Europe, Asia, or the Americas. The model breaks your sequence into overlapping 512-letter windows and votes across all of them.</div>
        <div class="eli5-step"><span class="eli5-num">3</span>Tab 2 lets you test a specific DNA mutation: change one letter at a specific position and the model predicts whether that change is likely harmful (pathogenic) or harmless (benign), using what it learned from thousands of known disease mutations.</div>
        <div class="eli5-step"><span class="eli5-num">4</span>Tab 3 converts your sequence into a 256-number fingerprint (an embedding) and shows you where it sits on a map of 100 reference human genomes. Sequences from the same family group cluster together — it's like a family portrait of human mitochondrial diversity.</div>
      </div>
    </div>`,
    math: `<div class="math-block">
      <div class="math-heading">k-NN interpolation for UMAP projection</div>
      <div class="math-eq">
        Given query embedding q and reference set {(e_i, x_i)} where x_i are pre-computed 2D coords:

        d_i = ||q - e_i||_2  (Euclidean distance in 256-dim space)
        Select top-k nearest: I = argsort(d)[0:k]
        w_i = 1/d_i  for i in I,  then  w = w / sum(w)  (normalise)
        x_query = sum(w_i * x_i  for i in I)  (weighted average 2D position)
      </div>
      <div class="math-example"><strong>Example:</strong> Query sequence has distances d=[0.8, 1.2, 2.1, 3.5, 4.0] to the 5 nearest reference points with 2D coords [(1,2), (1.5, 2.2), (3, 4), (2, 1), (0, 3)]. Weights w=[1/0.8, 1/1.2, 1/2.1, 1/3.5, 1/4.0] = [1.25, 0.83, 0.48, 0.29, 0.25] → normalised [0.39, 0.26, 0.15, 0.09, 0.08]. Projected x = 0.39*(1,2) + 0.26*(1.5,2.2) + ... ≈ (1.4, 2.2). The query appears near haplogroups H and HV on the UMAP.</div>
    </div>`
  };
})();
