# I Built a Pathogenicity Predictor Without the Evaluation Dataset

There's a model on HuggingFace called `vthawfeek/mtdna-fm-pathogenicity`. It has adapter weights. The training code ran. The architecture is correct.

I never evaluated it on real data, because I never built the evaluation dataset.

This post explains what the model does, why the evaluation gap exists, and what it would take to close it.

---

## The architecture

`MtDNAForVariantPathogenicity` wraps the pre-trained mtDNA-FM encoder with a binary classification head for variant pathogenicity prediction.

The key architectural decision: instead of using the `[CLS]` token representation, the model extracts the hidden state at the **variant position**. For a point mutation at position 3,243, it takes the embedding of the k-mer token that covers position 3,243 and projects it through a linear head to a pathogenic/benign score.

This is the right inductive bias. Pathogenicity is a local property — the effect of a mutation depends on what's in the immediate sequence context, not the global genome summary. `[CLS]` embeds the whole 16,569 bp genome. The variant token embeds the neighbourhood of the mutation.

```python
# Simplified: extract hidden state at variant position
hidden = encoder(input_ids, position_ids, attention_mask).last_hidden_state  # (B, L, D)
variant_hidden = hidden[torch.arange(B), variant_positions]                   # (B, D)
logits = self.classifier(variant_hidden)                                       # (B, 1)
```

---

## The training pipeline

LoRA r=4 fine-tuning on the pre-trained encoder. Binary cross-entropy with `pos_weight=2.5` to handle the imbalance between pathogenic and benign variants. A `PathogenicityVariantDataset` class that reads a parquet file with three required columns: `sequence` (the full 16,569 bp mtDNA context), `position` (0-indexed variant position), `label` (1=pathogenic, 0=benign).

This all works. The training loop runs. The model produces predictions.

---

## The missing piece: the evaluation dataset

A credible evaluation dataset for mtDNA pathogenicity requires two sources:

**Pathogenic variants:** ClinVar's "pathogenic" or "likely pathogenic" mtDNA variant records. These are human-curated disease-causing mutations with clinical evidence. The download client exists in this project (`mtdna_fm/data/variant_downloader.py`).

**Benign variants:** gnomAD's common mtDNA variants with allele frequency > 0.01. Common variants in the general population are strong evidence against severe pathogenicity. The gnomAD client also exists.

The steps to build the evaluation dataset:

1. Download ClinVar mtDNA pathogenic variants
2. Download gnomAD mtDNA common variants (AF > 0.01)
3. Map variant positions to the rCRS reference coordinate system (both databases use rCRS, but validation is required)
4. Reconstruct full-length sequences for each variant using the reference + alt allele
5. Deduplicate and balance the positive/negative sets
6. Create train/test splits (stratified by variant type: missense, nonsense, tRNA, rRNA)
7. Validate that no test variants appear in ClinVar's training annotations

Steps 1–2 would take a few hours. Steps 3–6 require a full day of careful data engineering. Step 7 requires understanding ClinVar's data model in more detail than I had time for during this project.

---

## Why this wasn't done

The project plan allocated most evaluation time to the haplogroup classifier (the most interpretable downstream task) and the zero-shot demonstrations (the most compelling results). The pathogenicity evaluation kept being deprioritised because it required data engineering before any modeling could happen.

The ordering error: I built the model architecture, training loop, and even trained and pushed the weights — before confirming that the evaluation data could be assembled in the time available. By the time I had a working training pipeline, the project was in its final week and preparing a clean ClinVar + gnomAD evaluation split wasn't feasible.

The lesson this teaches: the evaluation benchmark should be specified (and ideally assembled) before the model is written. Not because evaluation is more important than modeling, but because the evaluation defines what "done" means. Without it, you can finish training and still not know whether you built anything useful.

---

## What it would take to close the gap

A one-weekend effort:

1. Run `uv run python mtdna_fm/data/variant_downloader.py --source clinvar --output data/raw/clinvar_mtdna.csv`
2. Run the gnomAD equivalent
3. Write `scripts/prepare_variant_data.py` (50–100 lines of pandas) to join, clean, and split
4. Run `uv run mtdna-finetune --task pathogenicity --config configs/finetuning_pathogenicity.yaml`
5. Run `uv run mtdna-evaluate --model models/finetune_pathogenicity_paper --output-dir reports`

The infrastructure exists. The gap is step 3 — the data preparation script — which is straightforward but was never written.

---

## The architecture is still worth publishing

The variant-token hidden state approach for local mutation effect prediction is generalizable beyond mtDNA. The same architecture applies to any case where you want to predict the effect of a specific position change in a longer sequence context — splice-site variants, promoter mutations, protein stability changes from single amino acid substitutions.

The model exists. The evaluation doesn't. These are two separate problems, and only one of them is solved.
