# Scientific Review: mtDNA-FM Manuscript
**Reviewers:** Principal Biomedical Scientist + Machine Learning Scientist (Foundation Models)  
**Target venue:** bioRxiv preprint  
**Manuscript:** "mtDNA-FM: A Domain-Specialized Foundation Model for Mitochondrial DNA with Circular Positional Encoding and a Heteroplasmy Projection Channel"  
**Author:** Thawfeek Varusai  

---

## Executive Summary

This manuscript presents mtDNA-FM, a BERT-style encoder pre-trained on mitochondrial DNA sequences, with two architectural innovations: circular positional encoding (PE) and a heteroplasmy projection channel. The zero-shot evaluation results (37.9% haplogroup classification, AUROC 0.777 pathogenicity discrimination) are reproducible and honestly presented. The analysis section — which characterises the model's failure relative to DNABERT-2 using clade-specific per-class metrics — is scientifically rigorous and constitutes a genuine contribution.

The manuscript is **not yet ready for bioRxiv deposit** in its current form. Eight issues must be resolved before submission: two factual errors, one inverted inference, one misleading result presentation, two underpowered observations cited in main text without caveat, and two reproducibility gaps. Several additional low-effort clarifications will prevent predictable reviewer objections.

All claimed experimental results have been independently verified against stored JSON result files (`reports/zeroshot_haplogroup_knn.json`, `reports/zeroshot_pathogenicity_knn.json`, `reports/dnabert2_haplogroup_knn.json`), the codebase, and the MLflow database. Numbers match throughout.

---

## Part I: Scientific Contributions

The following contributions are genuine, correctly scoped, and appropriately caveated.

**1. Circular positional encoding (Contribution 1).**  
The mathematical formulation (φ_p = 2πp/L_genome) is correct. The ε ≈ 3.8×10⁻⁴ gap at the junction is verifiable. The implementation as a non-learnable `nn.Buffer` is appropriate and prevents gradient corruption of the topological prior during fine-tuning. **Critically: both the pre-training dataset class (`dataset.py:107`) and the inference API (`api.py:166`) construct windows with circular indexing (`indices = [(start+i) % n for i in range(window_size)]`), so the last window of each genome pass spans the D-loop boundary (positions 16,384–16,568 and 0–326 in a single 512-token forward pass).** Positions 0 and 16,568 are co-attended in every genome pass during both training and inference. The PE correctly assigns them nearly identical encodings at this joint. The circular PE claim is fully operationalised in the implementation. The paper does not describe this circular windowing mechanism, which causes the claim to appear weaker than it is.

**2. Two-phase domain-adaptive curriculum (Contribution 3).**  
Phase 1 (117,615 cross-species vertebrate sequences, 50k steps) followed by Phase 2 (34,975 human HmtDB sequences with heteroplasmy loss, 25k steps) is a coherent and biologically motivated curriculum. Phase 2 training was executed. MLflow DB records confirm Phase 1 reached a final MLM validation loss of 4.25 (perplexity 70) vs. a random baseline of 8.32 (perplexity 4,087), confirming the encoder learned non-trivial sequence representations. Phase 1 vs. Phase 2 ablation is appropriately deferred.

**3. Zero-shot evaluation with honest per-class analysis (Contribution 4).**  
The paper correctly reports that mtDNA-FM underperforms DNABERT-2 by 28.4 percentage points and provides a structured causal analysis across four observations. The clade-specific failure pattern (West Eurasian haplogroups F1 range 0.000–0.241 vs. African L haplogroups 0.438–0.857, no overlap across 15 class measurements) is a robust and interpretable finding. This analysis is the most scientifically valuable contribution of the manuscript.

**4. Heteroplasmy projection channel (Contribution 2).**  
The architectural design is novel. The channel is active in Phase 2 pre-training (het_weight=0.3). Empirical validation is correctly deferred.

**5. Public availability.**  
DVC pipeline, GitHub, HuggingFace weights, and Gradio demo are all available. This is commendable for a preprint.

---

## Part II: Critical Errors Requiring Correction Before bioRxiv Submission

### Error 1: Circular windowing mechanism is not described, making the PE claim appear unsupported

**Location:** §3.2 Pre-training, §3.3 Evaluation Setup, Abstract  
**Nature:** Omission that misrepresents the implementation  

The manuscript states the model uses "65 overlapping windows" without specifying that windowing is circular. A reader (and this reviewer, upon first pass) will assume linear windowing, conclude that positions 0 and 16,568 are never co-attended, and judge the circular PE claim as geometrically motivated but mechanistically inert. The implementation is more sophisticated: windows wrap around the D-loop boundary, directly validating the circular PE design.

**Required fix:** Add one sentence to §3.2 or the Evaluation Setup: *"Windows are constructed with circular indexing; the last window spans positions 16,384–16,568 and 0–326 in a single 512-token forward pass, ensuring positions 0 and 16,568 are co-attended in every genome pass during both pre-training and inference."*

---

### Error 2: DNABERT-2 pre-training corpus is misdescribed

**Location:** §1 Introduction (line ~98), §2.1 Related Work (line ~154), Supplementary S3.2  
**Nature:** Factual error  

The manuscript states DNABERT-2 was "pre-trained on 32 vertebrate nuclear genomes." The published paper (Zhou et al., ICLR 2024) describes pre-training on genomes from 32 *species* spanning multiple kingdoms, including bacteria, fungi, plants, and animals — not exclusively vertebrates. This matters: if DNABERT-2's training included bacterial and plant genomes, it has exposure to circular-genome sequences (bacterial chromosomes, chloroplasts), which modifies the claim about mtDNA-FM's unique suitability for circular molecules.

**Required fix:** Replace all three instances of "32 vertebrate nuclear genomes" with the correct description from the original paper. The title of the DNABERT-2 paper is "Efficient foundation model and benchmark for **multi-species genome**" — this phrasing should guide the replacement.

---

### Error 3: "European HV clade" is applied to haplogroups that do not belong to it

**Location:** §5.1 Observation 1, §5.3 Observation 3, Tables 2 and 3, Abstract, Conclusion  
**Nature:** Phylogenetic mislabeling — will be caught by any mtDNA expert  

The manuscript groups H, HV, J, K, T, U, V, W, X as the "European HV clade." This is phylogenetically wrong. In PhyloTree Build 17:  
- HV is a clade; H and V are daughters of HV ✓  
- **J and T** belong to the JT clade (macro-haplogroup R → JT branch) — not HV  
- **K** is a subclade of U (specifically U8b/K) — not HV  
- **U** is in the HV/JT/U macro-clade but is a sister of HV, not within it  
- **W** is a basal West Eurasian haplogroup (macro-haplogroup N → W)  
- **X** is a basal West Eurasian / Native American haplogroup (macro-haplogroup N → X)  

The scientifically correct grouping for these nine haplogroups is "West Eurasian haplogroups" or "European-associated haplogroups." Using "HV clade" conflates a phylogenetic clade with a geographic grouping.

**Required fix:** Replace all instances of "European HV clade" and "nine European HV haplogroups" with "nine West Eurasian haplogroups (H, HV, J, K, T, U, V, W, X)" throughout main text, tables, and supplementary. Approximately six replacements in main.tex, two in supplementary.tex.

---

### Error 4: tRNA AUPRC result is presented without the relative-lift context that reveals it as the worst-performing class

**Location:** §4.2 Variant Pathogenicity Prediction, Figure 4c caption  
**Nature:** Misleading framing of a legitimate result  

The manuscript reports: *"tRNA (n+=44, AUROC 0.718, AUPR 0.773; class-specific random AUPR = 0.677)"*. The absolute AUPRC of 0.773 appears impressive relative to overall AUPRC of 0.440. However, the class-specific relative lift is 0.773 / 0.677 = **1.14×** — the *smallest* relative gain of any variant class evaluated (overall: 0.440 / 0.220 = 2.0×; missense: 0.303 / (56/304) = ~1.64×). The high absolute AUPRC occurs because pathogenic tRNA variants constitute 68% of the tRNA class subset (44 pathogenic / 65 total), making the class-specific random AUPR already very high at 0.677. A reader who does not compute this will conclude tRNA is the best-performing class when it is the weakest by relative performance.

**Required fix:** Add after the tRNA report: *"representing a 1.14× lift over the class-specific random baseline — the smallest relative gain among evaluated variant classes."*

---

### Error 5: Data overlap inference direction is inverted in the Limitations section

**Location:** §6 Discussion, Limitations bullet "Evaluation–pre-training overlap"  
**Nature:** Incorrect logical inference  

Current text: *"Since DNABERT-2 had no such overlap, the 28.4 percentage-point gap should be interpreted as a conservative lower bound rather than a definitive measure of the true performance difference."*

This reasoning is backwards. If evaluation sequences appeared in Phase 1 pre-training (sequence memorisation risk), then mtDNA-FM's 37.9% may be **inflated**. Inflated mtDNA-FM performance means the true gap between DNABERT-2 (66.3%) and mtDNA-FM could be **larger** than 28.4 points. The gap is a lower bound on DNABERT-2's advantage, not a conservative lower bound on mtDNA-FM's performance.

**Required fix:** Replace with: *"Since some evaluation sequences may have appeared in Phase 1 pre-training, mtDNA-FM's 37.9% may be inflated by sequence-level memorisation. The 28.4 percentage-point gap should therefore be treated as a lower bound on DNABERT-2's true advantage; the actual gap on fully held-out sequences may be larger."*

---

### Error 6 (Reproducibility): No MLM pre-training convergence evidence

**Location:** §3.2 Pre-training (absent)  
**Nature:** Missing mandatory evidence for a foundation model paper  

The manuscript presents no MLM loss curve, final perplexity, or masked token accuracy from Phase 1 pre-training. For a language model paper, convergence evidence is the minimum required to demonstrate the model learned from training rather than remaining near random initialisation. Without it, the 37.9% zero-shot accuracy — which is also confounded by training/evaluation overlap — is the only evidence of learning.

**Data available:** The MLflow database (`mlflow.db`) records a final Phase 1 MLM validation loss of **4.25** (perplexity **70**), compared to the random baseline of **8.32** (perplexity ≈ **4,087** over the 4,102-token vocabulary). This is a 58× reduction in perplexity and constitutes clear convergence evidence.

**Required fix:** Add to §3.2: *"Phase 1 training converged to a final MLM validation loss of 4.25 (perplexity 70), compared to a random-baseline loss of 8.32 (perplexity $\approx$4,087 over the 4,102-token vocabulary), confirming that the encoder learned non-trivial sequence representations."*

---

### Error 7 (Reproducibility): Variant token extraction is unspecified

**Location:** §3.3 Evaluation Setup, Pathogenicity paragraph  
**Nature:** Reproducibility gap for a core reported result  

The manuscript states *"Each variant is embedded as the hidden state of the pre-trained encoder at the variant token position."* With stride-1 6-mer tokenization, each genomic position p appears as the start of one token (position_id = p) and as an internal nucleotide of up to five other tokens (positions p−1 through p−5). The specification "at the variant token position" is ambiguous: which of these up to six tokens is used?

**From code** (`mtdna_fm/inference/api.py:268–269`): the implementation selects the token whose `position_id` equals the 0-indexed variant position — i.e., the **6-mer starting at p**, covering positions p through p+5. The variant site is at the 5′ end of this k-mer.

**Required fix:** Add to §3.3: *"For each variant at 0-indexed genomic position p, the embedding is the final-layer hidden state of the 6-mer token starting at p (the token whose position\_id equals p, covering positions p to p+5 inclusive)."*

---

### Error 8: L5 F1=0.857 cited in main text Observation 1 without caveat (N=4)

**Location:** §5.1 Observation 1 — "All six African L haplogroups (L0–L5) fall in 0.438–0.857"  
**Nature:** Statistically unreliable result presented as a key finding  

The F1=0.857 for haplogroup L5 is based on N=4 test sequences — 3 correctly classified, 1 error. A single prediction flip changes this F1 by 0.25. This value anchors the upper end of the African L range cited in the main text and in Table 2. The range 0.438–0.857 would be 0.438–0.816 (L2) if L5 were excluded due to insufficient sample size. The supplementary table caption warns about N<10 classes, but that warning does not reach the main text where this result is actively used to establish "complete non-overlap" between clades.

**Required fix:** Change to: *"All six African L haplogroups fall in 0.438–0.857; note that L5 (F1=0.857) has only N=4 test sequences and should be interpreted with caution — excluding L5, the range is 0.438–0.816 (L2)."* The non-overlap claim remains valid either way (0.816 > 0.241 for the best West Eurasian haplogroup U), but the caveat is scientifically necessary.

---

### Error 9: Haplogroup E (N=4) included in Observation 4 without sufficient caveat

**Location:** §5.4 Observation 4 — "Three haplogroups show mtDNA-FM outperforming DNABERT-2: C (0.632 vs 0.611), F (0.716 vs 0.613), and E (0.400 vs 0.222)"  
**Nature:** Anecdotal N=4 result included in a design implication  

The F1 difference for haplogroup E (0.400 vs 0.222) is based on N=4 test sequences. With N=4, the difference could arise from 1 correct vs. 0 correct predictions. This should not be presented alongside haplogroup C (N=40) and F (N=36) as evidence for a design conclusion. The text correctly notes "test sets for these classes are small (C:40, F:36, E:4)" but does not apply appropriate weight to this caveat.

**Required fix:** Add: *"The result for haplogroup E (N=4 test sequences) is anecdotal and should not be interpreted independently. The pattern is most reliably supported by haplogroup C (N=40), which has an adequately sized test set."*

---

## Part III: Methodological Weaknesses

These are not blocking errors but require disclosure or minor correction.

**W1 — ClinVar curation criteria not specified.**  
The positive set (118 variants) combines "Pathogenic," "Likely pathogenic," and "Pathogenic/Likely pathogenic" ClinVar entries without describing the minimum review star threshold or handling of "conflicting interpretations." Add one sentence: *"ClinVar entries with review status 'Pathogenic', 'Likely pathogenic', or 'Pathogenic/Likely pathogenic' were included; entries with conflicting interpretations of pathogenicity were excluded."*

**W2 — AF>1% benign proxy may include haplogroup-defining positions.**  
In mtDNA, haplogroup-defining variants can reach AF>1% in the populations where that haplogroup is common (e.g., H-defining variant at position 750 is common in Europeans). These variants may be informationally linked to haplogroup identity, potentially confounding the pathogenicity signal. Acknowledge in the Limitations section.

**W3 — D-loop masking exclusion scope is ambiguous.**  
The text says "The D-loop homopolymeric C-tract (positions 303–315) is excluded from masking." It is not clear whether the entire D-loop is excluded (which would be scientifically problematic, as the D-loop carries the primary haplogroup-diagnostic variants) or only the 12-position C-tract. Clarify that only positions 303–315 are excluded and the remainder of the D-loop is included in the MLM objective.

**W4 — Balanced test set vs. natural distribution not distinguished.**  
The reported 37.9% accuracy is on a balanced test set (up to 40 sequences per class). On the natural class distribution, where haplogroup H constitutes ~14% of sequences, accuracy would differ substantially. The Methods correctly describes the balanced sampling; the Results section should add one sentence noting this distinction.

**W5 — Phase 2 and heteroplasmy channel listed as contributions without empirical outcomes.**  
Contributions 2 (heteroplasmy channel) and 3 (two-phase curriculum) are listed as numbered contributions in §1, but all reported results use the Phase 1 checkpoint. Both are implementation-complete design features with no reported empirical effect. Reframe them as "architectural design features and training objectives present in the implementation; empirical validation is deferred to the extended paper" to avoid overstating the preprint's empirical scope.

**W6 — Heteroplasmy channel degenerate-case description is mechanistically imprecise.**  
The text states: *"When h_p = 0 for all positions, the het channel contributes a constant offset that LayerNorm absorbs."* Inspection of the implementation (`embeddings.py:133–157`) reveals that `nn.Linear(1, hidden_size, bias=True)` is used. When h_p = 0, the output is the bias vector (not zero). The *inner* `het_norm` normalises this to a constant scaled vector. It is the *outer* `self.layer_norm(emb)` (the embedding-level LayerNorm, not the het-channel LayerNorm) that recenters this constant contribution per token. The end result — graceful degradation to standard sequence input — is correct, but the stated mechanism is wrong. Correct to: *"When h_p = 0, the het channel adds a constant offset (the normalised bias of the het projection) that is recentered by the embedding-level LayerNorm, so the model degrades gracefully to standard sequence input."*

---

## Part IV: Specific Textual Corrections

| Item | Location | Issue | Required change |
|------|----------|--------|-----------------|
| C1 | §3.3 Pathogenicity | ClinVar selection criteria | Add one sentence specifying review status inclusion criteria |
| C2 | §3.3 Pathogenicity | Entrez query in footnote | Move NCBI Entrez query string from footnote to main Methods text |
| C3 | §3.3 Pathogenicity | HmtDB vs NCBI source switch | Add: "the zero-shot evaluation corpus (NCBI) partially overlaps with Phase 1 pre-training (also NCBI) but not Phase 2 (HmtDB); overlap applies to Phase 1 representations only." |
| C4 | Fig. 3a caption | "50 haplogroups" vs 26-class task | Add: "(sub-haplogroup labels shown for visualization; the classification task uses the 26 major PhyloTree classes)" |
| C5 | §4.1, line ~343 | Forward reference to non-existent confusion matrix | Remove or qualify: "a full 26×26 confusion matrix from the 5-NN predictions is not reported here and cross-clade error rates have not been formally verified" — either add the confusion matrix to supplementary or delete this forward reference |
| C6 | Table 1 caption | ~400-word caption | Reduce to 3 sentences; move protocol detail to Methods |
| C7 | Throughout | "heteroplasmy" / "variant allele fraction" used interchangeably | Use "variant allele fraction" for the het-channel scalar input; reserve "heteroplasmy level" for the biological phenomenon |
| C8 | §5.1 Observation 1 | Macro-haplogroup R grouping | The nine West Eurasian haplogroups span multiple clades within macro-haplogroup R and N; consider specifying this when introducing the group |

---

## Part V: Items Appropriately Deferred — No Action Required

The following items are explicitly deferred to the extended paper and are correctly scoped for a preprint:

- Circular PE vs. sinusoidal PE ablation (mentioned 4 times as deferred)
- Heteroplasmy projection channel ablation
- Phase 1 vs. Phase 2 curriculum ablation
- Supervised fine-tuning of haplogroup classifier
- Pathogenicity benchmark against MitoTIP, APOGEE2, SIFT, PolyPhen-2
- Sub-haplogroup resolution (H1a vs. H1b etc.)
- Wet-lab experimental validation

The Discussion's treatment of circular PE generalizability to ecDNA, prokaryotic chromosomes, and ring chromosomes is speculative but appropriate for a Discussion section and does not require experimental support.

---

## Part VI: Summary Scorecard

| Category | Count | Blocking for bioRxiv? |
|----------|-------|----------------------|
| Critical errors requiring correction | 9 | Yes (all 9) |
| Methodological weaknesses | 6 | No (disclosure only) |
| Textual corrections | 8 | No (quality) |
| Deferred items (no action) | 7 | No |

**Verdict:** After applying all 9 critical corrections (particularly Errors 2, 3, 5 on factual accuracy; Errors 4, 8, 9 on statistical reliability; and Errors 6–7 on reproducibility), and the low-effort methodological disclosures, the manuscript will be ready for bioRxiv deposit. The core experimental claims are sound, reproducible, and honestly contextualized.

---

## Appendix: Verified Numerical Claims

All values below were independently verified against stored result files.

| Claim | Value | Source file | Status |
|-------|-------|-------------|--------|
| mtDNA-FM zero-shot haplogroup accuracy | 37.9% (95% CI 34.4–41.2%) | `reports/zeroshot_haplogroup_knn.json` | ✓ |
| DNABERT-2 zero-shot accuracy | 66.3% (95% CI 63.0–69.5%) | `reports/dnabert2_haplogroup_knn.json` | ✓ |
| Random baseline | 3.85% (1/26) | Analytical | ✓ |
| Pathogenicity AUROC | 0.777 (95% CI 0.731–0.821) | `reports/zeroshot_pathogenicity_knn.json` | ✓ |
| Pathogenicity AUPRC | 0.440 vs. 0.220 random | `reports/zeroshot_pathogenicity_knn.json` | ✓ |
| tRNA class-specific random AUPR | 44/65 = 0.677 | Calculated | ✓ |
| tRNA AUPRC lift | 0.773/0.677 = 1.14× | Calculated | ✓ |
| Total pre-training sequences | 152,590 | 117,615 + 34,975 | ✓ |
| Model parameters | ~5.8M | Architecture config | ✓ |
| MLM final perplexity (Phase 1) | 70 (loss 4.25) | `mlflow.db` eval runs | ✓ |
| MLM random baseline perplexity | ~4,087 (loss 8.32) | `mlflow.db` handsome-foal-37 run | ✓ |
| West Eurasian HV F1 range | 0.000–0.241 | `reports/zeroshot_haplogroup_knn.json` | ✓ |
| African L F1 range | 0.438–0.857 (L5 N=4) | `reports/zeroshot_haplogroup_knn.json` | ✓ |
| 6-mer LR supervised accuracy | 78.7% | `reports/kmer_baseline_haplogroup.json` | ✓ |
