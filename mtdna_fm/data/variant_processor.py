"""
mtDNA variant dataset processor.

Parses gnomAD chrM, ClinVar mtDNA, and PhyloTree Build 17 into clean
parquet files consumed by the fine-tuning tasks (Day 15+).

Three output files:
  variants_gnomad.parquet   — pos, ref, alt, af, het_level, n_het, n_hom
  variants_clinvar.parquet  — pos, ref, alt, label (1=pathogenic, 0=benign)
  haplogroup_markers.parquet— pos, ref, alt, haplogroup

All public functions are idempotent: they return immediately if the output
parquet already exists, so re-running the pipeline is safe.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GNOMAD_PARQUET = "variants_gnomad.parquet"
CLINVAR_PARQUET = "variants_clinvar.parquet"
HAPLOGROUP_PARQUET = "haplogroup_markers.parquet"

# ClinVar CLNSIG values treated as pathogenic
PATHOGENIC_CLNSIG: frozenset[str] = frozenset(
    [
        "Pathogenic",
        "Likely_pathogenic",
        "Pathogenic/Likely_pathogenic",
        "Pathogenic,_risk_factor",
    ]
)

# gnomAD AF threshold above which a variant is used as a benign proxy
BENIGN_AF_THRESHOLD: float = 0.01

# Regex for PhyloTree mutation strings: optional '!' + pos + ref + '>' + alt
# e.g. "3243A>G", "!16519T>C", "73A>G"
_MUTATION_RE = re.compile(r"^!?(\d+)([ACGTN])>([ACGTN])$", re.ASCII)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_info(info_str: str) -> dict[str, str]:
    """Split a VCF INFO string into a key→value dict."""
    result: dict[str, str] = {}
    for token in info_str.split(";"):
        if "=" in token:
            key, val = token.split("=", 1)
            result[key] = val
        elif token:
            result[token] = "true"
    return result


def _snp_only(ref: str, alt: str) -> bool:
    """Return True iff both alleles are single-base (SNP, not indel)."""
    return len(ref) == 1 and len(alt) == 1 and ref.upper() in "ACGT" and alt.upper() in "ACGT"


# ---------------------------------------------------------------------------
# gnomAD chrM VCF parser
# ---------------------------------------------------------------------------


def parse_gnomad_chrm_vcf(vcf_path: Path) -> pd.DataFrame:
    """Parse a gnomAD chrM VCF file into a variant DataFrame.

    Each PASS SNP becomes one row.  gnomAD chrM INFO fields used:

    * AF       — population allele frequency
    * mean_hl  — mean heteroplasmy level across heteroplasmic carriers
    * n_het    — count of heteroplasmic individuals
    * n_hom_var— count of homoplasmic alternate individuals

    If a field is absent, the corresponding column is NaN / 0.

    Returns
    -------
    DataFrame with columns: pos (int), ref (str), alt (str),
    af (float), het_level (float), n_het (int), n_hom (int).
    """
    records = []
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            _chrom, pos_s, _id, ref, alt, _qual, filt, info_s = parts[:8]
            if filt not in ("PASS", "."):
                continue
            if not _snp_only(ref, alt):
                continue
            info = _parse_info(info_s)
            try:
                af = float(info.get("AF", "nan"))
            except ValueError:
                af = float("nan")
            try:
                het_level = float(info.get("mean_hl", "nan"))
            except ValueError:
                het_level = float("nan")
            try:
                n_het = int(info.get("n_het", 0))
            except ValueError:
                n_het = 0
            try:
                n_hom = int(info.get("n_hom_var", info.get("n_hom", 0)))
            except ValueError:
                n_hom = 0
            records.append(
                {
                    "pos": int(pos_s),
                    "ref": ref.upper(),
                    "alt": alt.upper(),
                    "af": af,
                    "het_level": het_level,
                    "n_het": n_het,
                    "n_hom": n_hom,
                }
            )
    if not records:
        return pd.DataFrame(
            columns=["pos", "ref", "alt", "af", "het_level", "n_het", "n_hom"]
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ClinVar chrM VCF parser
# ---------------------------------------------------------------------------


def parse_clinvar_chrm_vcf(vcf_path: Path) -> pd.DataFrame:
    """Parse a ClinVar VCF (pre-filtered to chrM) → pathogenic variant DataFrame.

    Only PASS SNPs with CLNSIG matching PATHOGENIC_CLNSIG are kept.

    Returns
    -------
    DataFrame with columns: pos, ref, alt, label=1 (pathogenic only).
    Call :func:`add_benign_proxies` to add label=0 rows.
    """
    records = []
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            _chrom, pos_s, _id, ref, alt, _qual, _filt, info_s = parts[:8]
            if not _snp_only(ref, alt):
                continue
            info = _parse_info(info_s)
            clnsig = info.get("CLNSIG", "")
            # CLNSIG can be multi-valued with "|" or ","
            sigs = set(re.split(r"[|,/]", clnsig))
            if sigs & PATHOGENIC_CLNSIG:
                records.append(
                    {
                        "pos": int(pos_s),
                        "ref": ref.upper(),
                        "alt": alt.upper(),
                        "label": 1,
                    }
                )
    if not records:
        return pd.DataFrame(columns=["pos", "ref", "alt", "label"])
    return pd.DataFrame(records).drop_duplicates(subset=["pos", "ref", "alt"])


def add_benign_proxies(
    pathogenic_df: pd.DataFrame,
    gnomad_df: pd.DataFrame,
    af_threshold: float = BENIGN_AF_THRESHOLD,
) -> pd.DataFrame:
    """Augment a pathogenic DataFrame with benign proxies from gnomAD.

    A gnomAD variant is a benign proxy if:
    * AF >= af_threshold  (common enough to be presumed benign)
    * It does not appear in pathogenic_df

    Parameters
    ----------
    pathogenic_df : DataFrame with pos, ref, alt, label=1
    gnomad_df     : DataFrame from :func:`parse_gnomad_chrm_vcf`
    af_threshold  : minimum AF to treat as benign (default 0.01)

    Returns
    -------
    DataFrame with pos, ref, alt, label (1=pathogenic, 0=benign).
    """
    common = gnomad_df[gnomad_df["af"] >= af_threshold].copy()

    if not pathogenic_df.empty:
        path_keys = set(
            zip(pathogenic_df["pos"], pathogenic_df["ref"], pathogenic_df["alt"], strict=False)
        )
        mask = common.apply(
            lambda r: (r["pos"], r["ref"], r["alt"]) not in path_keys, axis=1
        )
        common = common[mask]

    benign = common[["pos", "ref", "alt"]].copy()
    benign["label"] = 0

    return pd.concat([pathogenic_df, benign], ignore_index=True)


# ---------------------------------------------------------------------------
# PhyloTree Build 17 parser
# ---------------------------------------------------------------------------


def parse_phylotree_csv(csv_path: Path) -> pd.DataFrame:
    """Parse a PhyloTree Build 17 CSV into a haplogroup marker DataFrame.

    Expected CSV format (header + data rows):

        haplogroup,mutation
        A,73A>G
        A,263A>G
        B,73A>G
        ...

    Mutation strings follow the PhyloTree convention:
      ``<pos><ref>><alt>``  e.g. ``3243A>G``

    A leading ``!`` means a back-mutation and is stripped (the variant is
    still recorded as a haplogroup marker because it defines the clade).

    Rows with unrecognised mutation format are silently dropped (common for
    insertions like ``315.1C`` which are not SNPs).

    Returns
    -------
    DataFrame with columns: pos (int), ref (str), alt (str), haplogroup (str).
    """
    df_raw = pd.read_csv(csv_path)
    if "haplogroup" not in df_raw.columns or "mutation" not in df_raw.columns:
        raise ValueError(
            f"PhyloTree CSV must have 'haplogroup' and 'mutation' columns, "
            f"got: {list(df_raw.columns)}"
        )
    records = []
    for _, row in df_raw.iterrows():
        m = _MUTATION_RE.match(str(row["mutation"]).strip())
        if m is None:
            continue
        pos, ref, alt = m.group(1), m.group(2), m.group(3)
        records.append(
            {
                "pos": int(pos),
                "ref": ref.upper(),
                "alt": alt.upper(),
                "haplogroup": str(row["haplogroup"]).strip(),
            }
        )
    if not records:
        return pd.DataFrame(columns=["pos", "ref", "alt", "haplogroup"])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Parquet builders (idempotent)
# ---------------------------------------------------------------------------


def build_gnomad_parquet(vcf_path: Path, output_dir: Path) -> Path:
    """Parse gnomAD chrM VCF and write variants_gnomad.parquet.

    Returns the output path. Skips parsing if the file already exists.
    """
    output_path = output_dir / GNOMAD_PARQUET
    if output_path.exists():
        logger.info("gnomAD parquet already exists: %s", output_path)
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    df = parse_gnomad_chrm_vcf(vcf_path)
    df.to_parquet(output_path, index=False)
    logger.info("gnomAD: wrote %d variants to %s", len(df), output_path)
    return output_path


def build_clinvar_parquet(
    vcf_path: Path,
    output_dir: Path,
    gnomad_parquet: Path | None = None,
) -> Path:
    """Parse ClinVar chrM VCF and write variants_clinvar.parquet.

    If *gnomad_parquet* is provided, common gnomAD variants are added as
    benign proxies (label=0).  Returns the output path.
    """
    output_path = output_dir / CLINVAR_PARQUET
    if output_path.exists():
        logger.info("ClinVar parquet already exists: %s", output_path)
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    df = parse_clinvar_chrm_vcf(vcf_path)
    if gnomad_parquet is not None and gnomad_parquet.exists():
        gnomad_df = pd.read_parquet(gnomad_parquet)
        df = add_benign_proxies(df, gnomad_df)
    df.to_parquet(output_path, index=False)
    logger.info("ClinVar: wrote %d labeled variants to %s", len(df), output_path)
    return output_path


def build_haplogroup_markers_parquet(csv_path: Path, output_dir: Path) -> Path:
    """Parse PhyloTree Build 17 CSV and write haplogroup_markers.parquet.

    Returns the output path. Skips parsing if the file already exists.
    """
    output_path = output_dir / HAPLOGROUP_PARQUET
    if output_path.exists():
        logger.info("Haplogroup markers parquet already exists: %s", output_path)
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    df = parse_phylotree_csv(csv_path)
    df.to_parquet(output_path, index=False)
    logger.info(
        "PhyloTree: wrote %d haplogroup markers to %s", len(df), output_path
    )
    return output_path
