# D-loop Entropy and Why mtDNA Preprocessing Isn't Trivial

The mitochondrial D-loop starts at position 576 of the 16,569-bp human genome and ends around position 16,024. That leaves 448 bases of coding sequence wrapping around the circular chromosome before the D-loop begins again. Computing Shannon entropy across tens of thousands of aligned sequences reveals something that textbooks state but doesn't fully land until you see it in your own data: the D-loop is roughly seven times more variable than the rest of the genome. Not twice as variable. Seven times.

That asymmetry shapes every decision in the Day 4 preprocessing pipeline.

## What "preprocessing" means here

Preprocessing for a DNA foundation model is not the same as normalising tabular features or resizing images. The raw FASTA files from HmtDB and NCBI contain sequences that differ in length (anywhere from 16,519 to 16,819 bp), character set (IUPAC ambiguity codes like R, Y, W, S, K, M alongside the standard ACGTN), and one quirk specific to circular genomes: some databases append the first 200 bases to the end of the sequence so that analysis tools running sliding windows don't miss the circular junction region.

The pipeline has five separate functions. Each is independently testable and independently importable. The decision to use functions rather than a preprocessing class is deliberate: a downstream consumer who only needs sequence cleaning doesn't need to touch length normalization, and a test for the split logic doesn't need real sequence data.

### Step 1: clean_sequence

```python
def clean_sequence(seq: str) -> str:
    seq = seq.upper()
    seq = re.sub(r"[^ACGTN]", "N", seq)
    n = JUNCTION_DUPLICATE_CHECK_BASES  # 200
    if len(seq) > 2 * n and seq[:n] == seq[-n:]:
        seq = seq[:-n]
    return seq
```

Uppercase is straightforward. The regex replacement catches all IUPAC ambiguity codes and other characters (hyphens from some alignment formats, dots from some annotation formats) and maps them to N. The junction duplicate detection compares the first 200 bases against the last 200: if they match exactly, the trailing copy is removed.

That last step is easy to overlook. If you leave junction duplicates in and then run length normalization, the trimming step removes real sequence from the 3' end rather than the appended duplicate. The sequences come out 16,569 bp as required, but the last ~150 bases of cytochrome b (positions 14,747-15,887) get replaced with what should have been trimmed away. Downstream: the model learns to predict junction sequence where coding sequence should be, and the loss at those positions is slightly lower because the training label is wrong.

### Step 2: normalize_length

Every sequence must be exactly 16,569 bp for batching. Sequences shorter than this get N padding inserted at position 576, not appended to the 3' end:

```python
def normalize_length(seq, target_length=16569, pad_position=576):
    if len(seq) < target_length:
        n_pad = target_length - len(seq)
        insert_at = min(pad_position, len(seq))
        seq = seq[:insert_at] + "N" * n_pad + seq[insert_at:]
    else:
        seq = seq[:target_length]
    return seq
```

The padding position matters for the same reason as the junction check: gene coordinate preservation. MT-CO1 (cytochrome c oxidase subunit 1) starts at position 5,904. If you pad by appending Ns at the 3' end, position 5,904 in the padded sequence still points to the same base as in the original. But if the original sequence is shorter than 576 bp and you insert padding at position 576, you shift everything after the insert point by `n_pad` positions. For sequences that are already close to 16,569 bp (which is most of them: the length histogram peaks within 50 bp of rCRS length), the insert is small and the displacement is minimal.

For sequences much shorter than rCRS (rare, but present in cross-species datasets), the D-loop region absorbs the padding. This is biologically appropriate: the D-loop is the control region, it is the most tolerant of insertions, and positioning the padding there rather than inside a protein-coding gene avoids teaching the model spurious patterns in functional regions.

### Step 3: stratified_split

The split uses `StratifiedShuffleSplit` from scikit-learn to produce 80/10/10 train/val/test partitions with proportional haplogroup representation in each. The implementation hits one non-obvious edge case: some haplogroups in the HmtDB corpus have only a single representative. `StratifiedShuffleSplit` requires at least 2 samples per class to estimate proportions. The fix: merge singleton classes into a `_rare` bin for the split calculation, then restore the original labels.

Cross-species sequences (NCBI vertebrate mtDNA) have no haplogroup label and go directly to train. They are used for Phase 1 MLM pre-training only and never appear in the val or test splits, which are evaluated on human sequences with known haplogroup labels.

## The D-loop entropy analysis

The EDA notebook computes per-position Shannon entropy across the corpus:

```python
bases = list("ACGTN")
freqs = np.stack([(seq_matrix == b).mean(axis=0) for b in bases], axis=0)
pos_entropy = np.array([scipy_entropy(freqs[:, pos] + 1e-9, base=2) for pos in range(RCRS_LENGTH)])
```

This is simple: at each genomic position, count the frequency of each of the five possible characters across all sequences, then compute the information-theoretic entropy of that frequency distribution. High entropy means many different bases appear at that position across different individuals; low entropy means most sequences agree on the same base.

The result: mean entropy in the D-loop is approximately 7x higher than in coding regions. The jump at position 576 is sharp. The protein-coding genes (MT-ND1 through MT-ND6, MT-CO1 through MT-CO3, MT-CYB, and the ATP synthase subunits) show entropy close to zero at most positions, reflecting strong evolutionary constraint. The D-loop shows broad high-entropy bands corresponding to the hypervariable regions (HV1 at positions 16,024-16,383 and HV2 at positions 73-340) that are used clinically for haplogroup assignment.

This asymmetry is one reason the preprocessing decisions above matter as much as they do. The model will see the D-loop region differently from the rest of the genome in terms of the token prediction task. Positions that are highly variable are harder to predict from context, which means they contribute more to the MLM loss and more to the gradient signal. If the D-loop boundary is blurred by incorrect padding or junction artifact, the model learns a smoothed version of the entropy landscape rather than the sharp boundary that actually exists.

## Why circular PE is the right response

The entropy analysis also shows why circular positional encoding matters beyond a vague "mtDNA is circular" argument. Consider the junction: positions 16,024 through 16,569 (the end of HV1 and the terminal coding sequence) and positions 1 through 576 (the beginning of HV2 and the tRNA cluster) are biologically contiguous. A model using standard BERT positional embeddings represents position 16,569 and position 1 as 16,568 positions apart, which is the furthest possible distance in the encoding space. A model using circular positional encoding, where the angle `2*pi*pos/genome_length` wraps continuously, represents them as one step apart.

The EDA shows that HV1 (ending near position 16,383) and HV2 (starting at position 73) have similar entropy profiles: both are high-variability regions used for haplogroup assignment. They are part of the same biological control region, separated only by the coordinate origin convention of the rCRS. A model that cannot represent their proximity cannot learn that they form a single functional unit.

## What the pipeline produces

The output schema for each parquet file:

| Column | Type | Notes |
|--------|------|-------|
| accession | string | NCBI or HmtDB sequence ID |
| sequence | string | cleaned, normalized to 16,569 bp |
| haplogroup | string/null | top-level haplogroup label; null for cross-species |
| species | string | "homo_sapiens" or vertebrate species name |
| geographic_origin | string/null | HmtDB geographic field where available |
| het_level_vector | null | populated from gnomAD on Day 5 |
| qc_pass | bool | True if N-content <= 10% |

The `het_level_vector` column is null at this stage. It will hold a float array of per-position heteroplasmy levels from gnomAD when gnomAD variant data is merged in on Day 5. The column exists in the schema now so the parquet files have a stable shape that downstream code can depend on.

## Numbers

- 67 unit tests passing (27 new preprocessor tests added today)
- 0 ruff errors across mtdna_fm/ and tests/
- All five preprocessor functions independently unit-tested
- Pipeline handles sequences from 16,519 bp (short cross-species) to 16,819 bp (with junction duplicate) without errors

The preprocessing pipeline is the last thing between raw data and the PyTorch Dataset class on Day 6. It needs to be correct before the model sees any training examples. Getting it right in the test suite now means the rest of the project can assume clean, 16,569-bp, all-uppercase, N-flagged sequences without defensive checks scattered through the training code.
