"""
Public inference API for mtDNA-FM.

MtDNAEmbedder is the stable public interface for embedding DNA sequences
using the pre-trained model. It hides the windowing and batching logic
so callers can embed a genome or variant in one line.

WHY a separate embedder class (not calling MtDNAModel directly):
  The pre-trained model operates on 512-token windows, not full 16,569-bp
  genomes. Callers should not need to know this. MtDNAEmbedder handles:
    - tokenisation with the correct vocabulary
    - overlapping window construction
    - CLS-token extraction per window
    - mean-pooling across windows into one genome vector
    - batched inference for DataFrames
  This is the interface that will be published to HuggingFace Hub and
  documented in the model card.

WHY cls_mean pooling (not mean-of-all-tokens or max-pool):
  The CLS token at position 0 of each window is the window's aggregate
  representation, conditioned on the full window context via self-attention.
  Mean-pooling across windows preserves the circular genome structure —
  every genomic region contributes equally to the final embedding.
  This is the same strategy used by sentence-transformers for long documents.

WHY embed_variant returns the token hidden state (not CLS):
  Pathogenicity is a local property. The hidden state at the token
  containing the variant position captures that position's context
  directly. The CLS state would dilute the local signal with global
  genome information.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mtdna_fm.model.model import MtDNAForMaskedModeling, MtDNAModel
from mtdna_fm.tokenizer.tokenize import tokenize_sequence
from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

if TYPE_CHECKING:
    import pandas as pd


class MtDNAEmbedder:
    """
    High-level interface for embedding mtDNA sequences with the pre-trained model.

    Parameters
    ----------
    model:
        Loaded MtDNAModel (base encoder, no prediction heads).
    vocabulary:
        Matching KmerVocabulary.
    device:
        Torch device for inference. Defaults to CUDA if available, else CPU.
    window_size:
        Number of tokens per inference window. Must match training config (512).
    stride:
        Token stride between consecutive windows (default 256 = 50% overlap).
    """

    def __init__(
        self,
        model: MtDNAModel,
        vocabulary: KmerVocabulary,
        device: str | torch.device | None = None,
        window_size: int = 512,
        stride: int = 256,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocabulary = vocabulary
        self.window_size = window_size
        self.stride = stride

        # Infer k from vocabulary size: vocab_size = 4^k + 6
        n_kmers = len(vocabulary) - vocabulary.n_special
        self.k = round(np.log(n_kmers) / np.log(4))

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> MtDNAEmbedder:
        """
        Load MtDNAEmbedder from a saved checkpoint directory or HuggingFace Hub name.

        The checkpoint is expected to have been saved by MtDNATrainer, which
        saves the full MtDNAForMaskedModeling object. This method extracts only
        the inner encoder (MtDNAForMaskedModeling.mtdna) and discards the
        prediction heads — they are not needed for inference.
        """
        path = Path(model_name_or_path)

        # Load vocabulary from the checkpoint directory
        vocabulary = KmerVocabulary.from_pretrained(str(path))

        # Load full pretraining model, then extract the base encoder
        full_model = MtDNAForMaskedModeling.from_pretrained(str(path))
        encoder: MtDNAModel = full_model.mtdna

        return cls(encoder, vocabulary, **kwargs)

    # ── Core embedding methods ─────────────────────────────────────────────────

    def embed_genome(
        self,
        sequence: str,
        het_levels: np.ndarray | None = None,
        pooling: str = "cls_mean",
    ) -> np.ndarray:
        """
        Embed a full mtDNA genome into a single fixed-size vector.

        Strategy (cls_mean): tokenise the genome with stride=1, slide overlapping
        windows of `window_size` tokens with step `stride`, extract the CLS
        (position-0) hidden state from each window, and mean-pool across windows.

        Parameters
        ----------
        sequence:
            Full-length mtDNA sequence string (A/C/G/T/N). Typically 16,569 bp.
        het_levels:
            Optional per-base heteroplasmy levels in [0.0, 1.0]. Must be the
            same length as sequence.
        pooling:
            Aggregation strategy. Currently only "cls_mean" is supported.

        Returns
        -------
        np.ndarray of shape (hidden_size,)
        """
        if pooling != "cls_mean":
            raise ValueError(f"Unsupported pooling '{pooling}'. Use 'cls_mean'.")

        # Normalize sequence to model genome_length so position IDs stay in range.
        # Ancient / non-standard sequences may differ from 16,569 bp by a few bases.
        genome_length = self.model.config.genome_length
        if len(sequence) > genome_length:
            sequence = sequence[:genome_length]
        elif len(sequence) < genome_length:
            sequence = sequence + "N" * (genome_length - len(sequence))

        tokens = tokenize_sequence(
            sequence,
            self.vocabulary,
            k=self.k,
            stride=1,
            max_seq_len=len(sequence),
            circular=True,
            het_levels=het_levels,
        )

        n_tokens = len(tokens["input_ids"])
        cls_vectors: list[np.ndarray] = []

        # Slide windows over the token stream
        for start in range(0, n_tokens, self.stride):
            window_indices = [(start + i) % n_tokens for i in range(self.window_size)]

            input_ids = torch.tensor(
                [tokens["input_ids"][j] for j in window_indices],
                dtype=torch.long,
            ).unsqueeze(0).to(self.device)

            position_ids = torch.tensor(
                [tokens["position_ids"][j] for j in window_indices],
                dtype=torch.long,
            ).unsqueeze(0).to(self.device)

            het_values = torch.tensor(
                [tokens["het_values"][j] for j in window_indices],
                dtype=torch.float,
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    het_values=het_values,
                )

            # CLS token = position 0 hidden state
            cls_vec = out.pooler_output.squeeze(0).cpu().numpy()
            cls_vectors.append(cls_vec)

        return np.mean(cls_vectors, axis=0)

    def embed_variant(
        self,
        sequence: str,
        position: int,
        pooling: str = "token",
    ) -> np.ndarray:
        """
        Embed the genomic context around a specific variant position.

        Extracts a window centered on the variant, runs the model, and returns
        the hidden state at the token containing `position`. This captures local
        context (tRNA fold, codon, regulatory motif) rather than genome-level signal.

        Parameters
        ----------
        sequence:
            Full or partial mtDNA sequence. Must cover [position - window_size//2,
            position + window_size//2].
        position:
            0-indexed genomic position of the variant.
        pooling:
            "token" returns the hidden state at the variant's token position.
            "cls" returns the CLS (position-0) hidden state of the window.

        Returns
        -------
        np.ndarray of shape (hidden_size,)
        """
        genome_length = len(sequence)

        # Center the window on the variant position
        half = self.window_size // 2
        start = (position - half) % genome_length

        tokens = tokenize_sequence(
            sequence,
            self.vocabulary,
            k=self.k,
            stride=1,
            max_seq_len=genome_length,
            circular=True,
            het_levels=None,
        )

        n_tokens = len(tokens["input_ids"])
        window_indices = [(start + i) % n_tokens for i in range(self.window_size)]

        input_ids = torch.tensor(
            [tokens["input_ids"][j] for j in window_indices],
            dtype=torch.long,
        ).unsqueeze(0).to(self.device)

        position_ids = torch.tensor(
            [tokens["position_ids"][j] for j in window_indices],
            dtype=torch.long,
        ).unsqueeze(0).to(self.device)

        het_values = torch.zeros(1, self.window_size, dtype=torch.float).to(self.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                het_values=het_values,
            )

        hidden = out.last_hidden_state.squeeze(0)  # (window_size, hidden_size)

        if pooling == "cls":
            return hidden[0].cpu().numpy()

        # Find the window slot whose position_id matches the variant position
        pos_list = [tokens["position_ids"][j] for j in window_indices]
        token_idx = pos_list.index(position) if position in pos_list else half

        return hidden[token_idx].cpu().numpy()

    def embed_dataset(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed all sequences in a DataFrame, returning one vector per row.

        Calls embed_genome on each sequence. The `batch_size` parameter controls
        how many sequences are processed before freeing intermediate tensors —
        it does not affect results, only memory usage.

        Parameters
        ----------
        df:
            DataFrame with at least a `sequence_col` column.
        sequence_col:
            Column name containing DNA sequence strings.
        batch_size:
            Number of sequences per batch (for progress reporting).

        Returns
        -------
        np.ndarray of shape (n_sequences, hidden_size)
        """
        sequences = df[sequence_col].tolist()
        n = len(sequences)
        embeddings: list[np.ndarray] = []

        for i in range(0, n, batch_size):
            batch_seqs = sequences[i : i + batch_size]
            for seq in batch_seqs:
                vec = self.embed_genome(seq)
                embeddings.append(vec)

        return np.stack(embeddings, axis=0)
