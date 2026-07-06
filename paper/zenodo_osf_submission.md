# Zenodo / OSF Preprints Submission: mtDNA-FM

Prepared as a parallel/fallback route while bioRxiv's independent-researcher affiliation
question is unresolved (see `paper/cover_letter.md` for the bioRxiv submission, which was
rejected for lack of institutional affiliation — MS ID# BIORXIV/2026/734309).

---

## Revision history

**rev2 — commit `fc47123` (pushed to `origin/main`) — the version to upload.**
This is the current, polished submission package. Changes since the bioRxiv-submitted PDF:

- **Code license standardized on Apache 2.0** (added repo-root `LICENSE`; fixed the manuscript's
  Code Availability wording from "MIT" — see the section below for the full rationale).
- **Title shortened** to *"...for Mitochondrial DNA with Circular Positional Encoding"* — dropped
  "and a Heteroplasmy Projection Channel," which the preprint does not evaluate. The channel is
  still described in the abstract and body as an architectural contribution, honestly hedged.
  **When filling the Zenodo/OSF title field, use the shortened title in the metadata section below.**
- **Circular-PE priority claim narrowed** from "first application to biological sequences" to a
  defensible claim scoped to genomic foundation models, acknowledging periodic sinusoidal/rotary
  PE. No new citations were fabricated; a verified literature search for cyclic/periodic-PE prior
  art is still an open follow-up (worthwhile before a journal, optional for the preprint).
- **Supplementary §S6** renamed "Ablation Learning Curves" → "Ablations (Deferred)" (it contained
  no curves).
- Both PDFs recompiled; the `paper/zenodo/` copies were refreshed from the recompiled source.

**rev1 — the bioRxiv-submitted `BIORXIV-2026-734309v1-Varusai.pdf`.** Frozen historical snapshot.
Still carries the old "MIT" license wording and the longer title. **Do not upload this file** —
it is superseded by rev2.

Not yet done (tracked separately, not blocking this upload): eval/pre-training overlap
quantification, and the circular-vs-sinusoidal PE ablation. See the project notes for effort/value.

---

## ✅ Code license — fixed

The manuscript, README, and repo used to disagree on the code license (MIT in the manuscript's
Code Availability section vs. Apache 2.0 on the README badge vs. no `LICENSE` file at all).
Resolved by standardizing on **Apache 2.0**, since that's what was already live on the public
HuggingFace model card (`mtdna_fm/scripts/push_to_hub.py:29`) — the manuscript text was the
only outlier and the only one not yet locked into a published artifact.

- Added `LICENSE` (Apache License 2.0, copyright Thawfeek Varusai, 2026) at the repo root.
- Fixed `paper/manuscript/main.tex` Code Availability section: "MIT license" → "Apache License 2.0".
- Recompiled `main.pdf` from the fixed source and verified via `pdftotext` that it now reads
  "Apache License 2.0."
- README badge already said Apache 2.0 — no change needed there.

This means the originally bioRxiv-submitted `BIORXIV-2026-734309v1-Varusai.pdf` still contains
the old, incorrect "MIT" wording (that's a frozen historical snapshot — it was rejected for the
affiliation reason, not this). **Do not upload that file to Zenodo/OSF.** Use the freshly
recompiled copies below instead, which have the correct license text.

---

## 📦 What to submit to Zenodo — step by step

1. Go to https://zenodo.org/deposit/new and sign in (ORCID login recommended so it links automatically).
2. Upload type: **Publication** → **Preprint**.
3. Upload these two files (already prepared, license-corrected):
   - [`paper/zenodo/mtDNA-FM_manuscript_v1.pdf`](zenodo/mtDNA-FM_manuscript_v1.pdf) — main manuscript
   - [`paper/zenodo/mtDNA-FM_supplementary_v1.pdf`](zenodo/mtDNA-FM_supplementary_v1.pdf) — supplementary materials
4. Fill in the metadata fields from the "Metadata common to both platforms" and "Zenodo-specific fields" sections below (title, author, abstract, keywords, license, related identifiers).
5. Click **Publish**. Zenodo mints a DOI immediately (`10.5281/zenodo.XXXXXXX`) — there is no review/waiting period, unlike bioRxiv.
6. Do the "After upload" steps at the bottom of this doc.

That's the whole Zenodo path — everything else in this file is reference detail for step 4.

---

## Metadata common to both platforms

**Title:**
mtDNA-FM: A Domain-Specialized Foundation Model for Mitochondrial DNA with Circular
Positional Encoding

**Author:**
- Name: Thawfeek Varusai
- Affiliation: Independent Researcher
- ORCID: [0000-0002-7864-5971](https://orcid.org/0000-0002-7864-5971)
- Email: vthawfeek@gmail.com

**Abstract** (plain text, ready to paste):

> Mitochondrial DNA (mtDNA) is a 16,569 bp circular genome with central roles in human
> disease, population genetics, and evolutionary biology. Existing DNA foundation models are
> pre-trained on nuclear genomes and use linear positional encodings, which misrepresent the
> circular topology of mtDNA: positions 0 and 16,568 are physically adjacent at the D-loop
> junction but receive maximally different encodings.
>
> We present mtDNA-FM, the first foundation model dedicated to mitochondrial DNA. It
> introduces two domain-specific design choices: (i) a circular positional encoding where
> positions 0 and 16,568 receive near-identical encodings (ε ≈ 3.8×10⁻⁴ per dimension),
> correctly representing their adjacency at the D-loop; and (ii) a heteroplasmy projection
> channel that accepts a continuous per-position variant allele fraction alongside sequence
> content. The model is pre-trained on 152,590 vertebrate mtDNA sequences in two phases
> (117,615 cross-species, then 34,975 human HmtDB).
>
> Zero-shot evaluation of the Phase 1 checkpoint achieves 37.9% on 26-class haplogroup
> classification (95% CI 34.4-41.2%; random baseline 3.85%; 9.9x lift) and AUROC 0.777 (95%
> CI 0.731-0.821) on pathogenic variant discrimination without labels. Comparison with
> DNABERT-2 (66.3%, 117M parameters) reveals a clade-specific gap in West Eurasian
> haplogroups. Per-class analysis yields four observations that identify specific
> representation failures and map concrete improvements for a next-generation model.
>
> Availability: https://github.com/vthawfeek/mtdna-foundation-model,
> https://huggingface.co/vthawfeek/mtdna-foundation-model,
> https://huggingface.co/spaces/vthawfeek/mtdna-fm-demo

**Keywords / tags:**
mitochondrial DNA; foundation model; transformer; DNA language model; positional encoding;
heteroplasmy; genomics; bioinformatics; haplogroup classification; pathogenic variant
prediction; self-supervised learning; zero-shot learning

**Manuscript license:** CC-BY 4.0 (matches what was planned for bioRxiv — maximum reuse,
standard for preprints)

**Related links:**
- Code: https://github.com/vthawfeek/mtdna-foundation-model
- Model weights: https://huggingface.co/vthawfeek/mtdna-foundation-model
- Demo: https://huggingface.co/spaces/vthawfeek/mtdna-fm-demo

**Data Availability statement** (from `main.tex`, paste as-is):
> Training data (HmtDB, NCBI Entrez) and variant databases (gnomAD, ClinVar) are publicly
> available at their respective sources. No proprietary or restricted data were used.
> Pre-trained model weights are available at
> https://huggingface.co/vthawfeek/mtdna-foundation-model.

**Conflict of interest:** None.

**Language:** English
**Version:** 1 (first public release — no prior git tag exists; fine to leave as "1" or "v1.0")
**Publication date:** date of actual upload

---

## Zenodo-specific fields

- **Upload type:** Publication
- **Publication type:** Preprint
- **Related/alternate identifiers** (add all three, relation type "Is supplemented by", identifier type URL):
  - `https://github.com/vthawfeek/mtdna-foundation-model`
  - `https://huggingface.co/vthawfeek/mtdna-foundation-model`
  - `https://huggingface.co/spaces/vthawfeek/mtdna-fm-demo`
- **Grants/Funding:** none
- **Community:** optional — skip unless you want to join a specific Zenodo community (e.g. a bioinformatics one), not required for the DOI
- Zenodo mints the DOI automatically on publish, format `10.5281/zenodo.XXXXXXX`

**Separate, optional step:** if you also want a DOI for the *code repository itself* (distinct
from the manuscript PDF), Zenodo's GitHub integration can auto-archive a GitHub Release. That
needs a `.zenodo.json` at the repo root and is independent of this manuscript upload — not
needed to get the preprint out.

## OSF Preprints-specific fields

- **Files:** same two files as Zenodo — `paper/zenodo/mtDNA-FM_manuscript_v1.pdf` and `paper/zenodo/mtDNA-FM_supplementary_v1.pdf`
- **Preprint service:** OSF Preprints (the generic multidisciplinary server — there's no
  dedicated bio-branded OSF server the way bioRxiv is dedicated; OSF Preprints accepts any field)
- **Subject area(s):** Life Sciences > Genetics and Genomics; also tag Computer and Information
  Science > Artificial Intelligence and Robotics if the interface allows multiple subjects
- **License:** CC-BY 4.0
- **Contributor:** Thawfeek Varusai (Admin), link ORCID via account profile so it syncs automatically
- **Has this been published elsewhere?** No
- **Conflict of interest:** None
- OSF mints a DOI on making the preprint public, format `10.31219/osf.io/XXXXX`

---

## After upload (either platform)

1. Add the resulting DOI to the HuggingFace model card and the GitHub README.
2. If your ORCID is linked to the Zenodo/OSF account, the work should auto-import to your ORCID record — verify it shows up.
3. Commit the new `LICENSE` file, the `main.tex` fix, and the recompiled `main.pdf` (see git status) — these haven't been committed yet.
