"""
VariantDataset for mtDNA pathogenicity fine-tuning.

For each SNP in the variant DataFrame, applies the alternate allele to the
rCRS reference sequence, tokenises the full genome, and returns a window of
`window_size` tokens centred on the variant position.  Only SNPs are
supported (single-base ref and alt).

Position convention: `pos` in the variant DataFrame is 1-based (VCF standard).
Internally it is converted to 0-based (Python indexing) before applying the SNP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from mtdna_fm.tokenizer.tokenize import tokenize_sequence
from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

if TYPE_CHECKING:
    import pandas as pd


class VariantDataset(Dataset):
    """
    Dataset for SNP-level pathogenicity classification.

    For each variant: applies the alt allele to the reference at
    position (pos - 1), tokenises the resulting sequence, then returns a
    `window_size`-token window centred on the variant token.

    The variant token hidden state (not CLS) is the natural input to a
    pathogenicity head because pathogenicity is a local property of the
    variant's sequence context.

    Parameters
    ----------
    reference:
        rCRS reference sequence string (genome_length bases).
    variants_df:
        DataFrame with columns: pos (int, 1-based VCF), ref (str),
        alt (str), label (int, 1=pathogenic, 0=benign).
    vocabulary:
        KmerVocabulary instance.
    k:
        K-mer size matching the vocabulary.
    window_size:
        Number of tokens in each returned window (default 512).
    genome_length:
        Reference genome length (default 16,569).
    """

    def __init__(
        self,
        reference: str,
        variants_df: pd.DataFrame,
        vocabulary: KmerVocabulary,
        k: int = 6,
        window_size: int = 512,
        genome_length: int = 16569,
    ) -> None:
        self.reference = reference.upper()
        self.vocabulary = vocabulary
        self.k = k
        self.window_size = window_size
        self.genome_length = genome_length

        # Keep only SNPs; reset index so iloc is contiguous
        mask = (variants_df["ref"].str.len() == 1) & (variants_df["alt"].str.len() == 1)
        self._variants = variants_df[mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._variants)

    def __getitem__(self, idx: int) -> dict:
        row = self._variants.iloc[idx]

        # VCF positions are 1-based; convert to 0-based
        pos_0 = int(row["pos"]) - 1

        # Apply SNP to reference
        seq_list = list(self.reference)
        seq_list[pos_0] = str(row["alt"]).upper()
        mutated_seq = "".join(seq_list)

        # Tokenise full sequence (stride=1, circular=True)
        tokens = tokenize_sequence(
            mutated_seq,
            self.vocabulary,
            k=self.k,
            stride=1,
            max_seq_len=self.genome_length,
            circular=True,
        )

        n = len(tokens["input_ids"])

        # Centre window on the variant token (token index == pos_0 for stride=1)
        half = self.window_size // 2
        window_start = (pos_0 - half) % n
        indices = [(window_start + i) % n for i in range(self.window_size)]

        # Offset of the variant token within the returned window
        variant_offset = half if pos_0 >= half else (pos_0 - window_start) % n

        return {
            "input_ids": torch.tensor([tokens["input_ids"][j] for j in indices], dtype=torch.long),
            "attention_mask": torch.tensor(
                [tokens["attention_mask"][j] for j in indices], dtype=torch.long
            ),
            "position_ids": torch.tensor(
                [tokens["position_ids"][j] for j in indices], dtype=torch.long
            ),
            "het_values": torch.tensor(
                [tokens["het_values"][j] for j in indices], dtype=torch.float
            ),
            "variant_pos": torch.tensor(pos_0, dtype=torch.long),
            "variant_offset": torch.tensor(variant_offset, dtype=torch.long),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }
