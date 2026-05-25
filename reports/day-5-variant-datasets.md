# Day 5: Variant Datasets

## What was built

- `mtdna_fm/data/variant_processor.py` — six functions for parsing and writing the three variant parquets:
  - `parse_gnomad_chrm_vcf`: reads a gnomAD chrM VCF, returns pos/ref/alt/af/het_level/n_het/n_hom per PASS SNP
  - `parse_clinvar_chrm_vcf`: reads a ClinVar chrM VCF, returns pathogenic SNPs (label=1)
  - `add_benign_proxies`: augments pathogenic DataFrame with gnomAD common variants (AF ≥ 0.01, label=0)
  - `parse_phylotree_csv`: parses PhyloTree Build 17 CSV (haplogroup, mutation columns) into pos/ref/alt/haplogroup
  - `build_gnomad_parquet`, `build_clinvar_parquet`, `build_haplogroup_markers_parquet` — idempotent builders that skip if output exists
- `mtdna_fm/data/variant_downloader.py` — three idempotent download functions:
  - `download_gnomad_chrm`: downloads .bgz + .tbi, extracts chrM VCF with tabix (or keeps .bgz if tabix unavailable)
  - `download_clinvar_chrm`: downloads GRCh38 ClinVar VCF (gzipped), filters for chrM in Python
  - `download_phylotree`: downloads PhyloTree Build 17 CSV from canonical URL
- `mtdna_fm/scripts/download.py` — replaced three `typer.Exit(code=1)` stubs (gnomad, clinvar, phylotree) with real `_run_gnomad`, `_run_clinvar`, `_run_phylotree` dispatch functions
- `tests/test_data.py` — 22 new tests across 4 classes:
  - `TestGnomadParser` (5): PASS SNP parsing, non-PASS skipped, indels skipped, empty VCF, schema columns
  - `TestClinvarParser` (5): pathogenic label=1, VUS/benign skipped, benign proxies added, no duplication, schema columns
  - `TestPhylotreeParser` (5): standard mutation string, back-mutation `!` prefix, non-SNP mutations skipped, wrong columns raises, schema columns
  - `TestBuildParquets` (5): gnomAD parquet created, idempotency, ClinVar parquet, haplogroup parquet, ClinVar with benign proxies
  - Plus 2 updated CLI tests replacing the old `test_unimplemented_source_exits_with_error` with working gnomad/clinvar/phylotree routing tests (3 new CLI tests total)

## What was learned

- **gnomAD chrM INFO fields differ from autosomal**: chrM uses `mean_hl` (mean heteroplasmy level) and `n_hom_var` (not `n_hom`) in its INFO column. Parsing must be tolerant of absent fields — returning NaN rather than crashing on missing INFO keys.
- **ClinVar CLNSIG is multi-valued**: the CLNSIG field uses `|` and `,` as delimiters within a single value (e.g., `Pathogenic|other`). A regex split on both delimiters is needed before testing set membership.
- **Benign proxy construction is a design decision**: using gnomAD AF ≥ 0.01 as a benign proxy is the standard approach in pathogenicity prediction, but it introduces a survivorship bias (very common variants that happen to be pathogenic in specific haplogroup contexts are mislabeled). The 0.01 threshold is a hyperparameter worth tuning at fine-tuning time.
- **PhyloTree mutation strings are heterogeneous**: Build 17 includes insertions (e.g., `315.1C`), deletions, and back-mutations (`!16519T>C`). A strict regex for `pos + ref + > + alt` cleanly separates usable SNPs from everything else without special-casing each variant type.
- **Tabix vs Python filtering trade-off**: tabix is the right tool for indexed VCF extraction (gnomAD .bgz is gigabytes), but ClinVar is small enough that Python's gzip line-by-line filter is simpler, faster to implement, and avoids a system dependency in tests.

## Key decisions

- **Standalone functions, not a class**: matches the Day 4 preprocessor pattern — each function is independently testable and importable. A future pipeline that only needs pathogenicity labels can import `parse_clinvar_chrm_vcf` without pulling in the gnomAD parser.
- **Idempotency on output parquets (not input VCFs)**: the builder functions check whether the output parquet exists, not whether the raw VCF has changed. This is intentional — re-parsing a VCF that hasn't changed is wasteful, and variant databases update on their own schedule.
- **benign AF threshold = 0.01**: matches gnomAD's own "common variant" definition. Made a named constant (`BENIGN_AF_THRESHOLD`) so it can be overridden at call time and is visible in the API surface rather than buried in a magic number.
- **`het_level_vector` column deferred to Day 5 parquet merge**: the preprocessed sequence parquet has a null `het_level_vector` column (added in Day 4 schema). Populating it requires joining the gnomAD variant parquet on position after both are built — that join happens in the PyTorch Dataset class on Day 6, not in the downloader.

## Verification

```
$ uv run ruff check mtdna_fm/ tests/
All checks passed!

$ uv run pytest tests/ -m "not slow and not integration" -q
89 passed in 3.14s
```

Spot-checks on the new functions:

```python
from mtdna_fm.data.variant_processor import (
    parse_gnomad_chrm_vcf, parse_clinvar_chrm_vcf,
    parse_phylotree_csv, add_benign_proxies
)

# gnomAD PASS SNP parsing
import io, pandas as pd
vcf_text = "##VCF\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchrM\t73\t.\tA\tG\t.\tPASS\tAF=0.987;mean_hl=0.99;n_het=12345;n_hom_var=55000\n"
# (written to temp file)
# → df["af"].iloc[0] == 0.987, df["n_het"].iloc[0] == 12345

# PhyloTree mutation string parsing
import pandas as pd, tempfile, pathlib
rows = [{"haplogroup": "A", "mutation": "73A>G"}, {"haplogroup": "A", "mutation": "315.1C"}]
# → only 73A>G parsed; 315.1C skipped (non-SNP)

# benign proxy deduplication
path = pd.DataFrame({"pos":[73],"ref":["A"],"alt":["G"],"label":[1]})
gnomad = pd.DataFrame({"pos":[73],"ref":["A"],"alt":["G"],"af":[0.987],"het_level":[0.99],"n_het":[1000],"n_hom":[50000]})
result = add_benign_proxies(path, gnomad)
assert len(result) == 1 and result.iloc[0]["label"] == 1  # 73 not duplicated
```

## Next up

Day 6: PyTorch Dataset class — `MtDNADataset` with overlapping 512-token windows (stride 256) over the 16,569-bp preprocessed sequences, and `VariantDataset` for SNP-centered pathogenicity windows.
