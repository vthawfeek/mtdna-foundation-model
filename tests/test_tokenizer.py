"""
Tests for KmerVocabulary and tokenize_sequence.
"""

import numpy as np
import pytest

from mtdna_fm.tokenizer import (
    CLS_TOKEN_ID,
    HET_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
    UNK_TOKEN_ID,
    KmerVocabulary,
    tokenize_sequence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vocab() -> KmerVocabulary:
    return KmerVocabulary.build(k=6)


@pytest.fixture(scope="module")
def vocab3() -> KmerVocabulary:
    return KmerVocabulary.build(k=3)


# ---------------------------------------------------------------------------
# KmerVocabulary tests
# ---------------------------------------------------------------------------


class TestKmerVocabulary:
    def test_vocabulary_size(self, vocab: KmerVocabulary) -> None:
        assert len(vocab) == 4102  # 4096 6-mers + 6 special tokens

    def test_vocabulary_size_k3(self, vocab3: KmerVocabulary) -> None:
        assert len(vocab3) == 64 + 6  # 64 3-mers + 6 special tokens

    def test_special_token_ids(self, vocab: KmerVocabulary) -> None:
        assert vocab.encode("[PAD]") == PAD_TOKEN_ID == 0
        assert vocab.encode("[CLS]") == CLS_TOKEN_ID == 1
        assert vocab.encode("[MASK]") == MASK_TOKEN_ID == 2
        assert vocab.encode("[UNK]") == UNK_TOKEN_ID == 3
        assert vocab.encode("[SEP]") == SEP_TOKEN_ID == 4
        assert vocab.encode("[HET]") == HET_TOKEN_ID == 5

    def test_kmer_ids_start_at_six(self, vocab: KmerVocabulary) -> None:
        # Smallest lexicographic 6-mer is AAAAAA, should be index 6
        assert vocab.encode("AAAAAA") == 6

    def test_encode_decode_roundtrip(self, vocab: KmerVocabulary) -> None:
        for kmer in ["ATGCAT", "GCTAGC", "TTTTTT", "ACGTAC"]:
            assert vocab.decode(vocab.encode(kmer)) == kmer

    def test_unknown_kmer_maps_to_unk(self, vocab: KmerVocabulary) -> None:
        assert vocab.encode("NNNNNN") == UNK_TOKEN_ID
        assert vocab.encode("ZZZZZZ") == UNK_TOKEN_ID

    def test_vocabulary_is_deterministic(self) -> None:
        v1 = KmerVocabulary.build(k=6)
        v2 = KmerVocabulary.build(k=6)
        assert v1.encode("ACGTAC") == v2.encode("ACGTAC")
        assert v1.encode("GCATGC") == v2.encode("GCATGC")

    def test_vocabulary_save_load(self, vocab: KmerVocabulary, tmp_path) -> None:
        vocab.save_pretrained(tmp_path)
        loaded = KmerVocabulary.from_pretrained(tmp_path)
        assert len(loaded) == len(vocab)
        assert loaded.encode("ATGCAT") == vocab.encode("ATGCAT")
        assert loaded.encode("[CLS]") == CLS_TOKEN_ID

    def test_unk_token_id_property(self, vocab: KmerVocabulary) -> None:
        assert vocab.unk_token_id == UNK_TOKEN_ID == 3

    def test_sep_token_id_property(self, vocab: KmerVocabulary) -> None:
        assert vocab.sep_token_id == SEP_TOKEN_ID == 4

    def test_het_token_id_property(self, vocab: KmerVocabulary) -> None:
        assert vocab.het_token_id == HET_TOKEN_ID == 5

    def test_contains(self, vocab: KmerVocabulary) -> None:
        assert "ACGTAC" in vocab
        assert "ZZZZZZ" not in vocab
        assert "[PAD]" in vocab


# ---------------------------------------------------------------------------
# tokenize_sequence tests
# ---------------------------------------------------------------------------


class TestTokenizeSequence:
    def test_circular_junction_covered(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 25  # 100 bp
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True)
        assert len(tokens["input_ids"]) == len(seq)

    def test_linear_token_count(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 25  # 100 bp
        k = 6
        tokens = tokenize_sequence(seq, vocabulary=vocab, k=k, circular=False)
        expected = len(seq) - k + 1  # 95
        assert len(tokens["input_ids"]) == expected

    def test_circular_stride2(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 25  # 100 bp
        tokens = tokenize_sequence(seq, vocabulary=vocab, stride=2, circular=True)
        assert len(tokens["input_ids"]) == 50  # 100 / stride=2

    def test_output_keys(self, vocab: KmerVocabulary) -> None:
        seq = "ACGTACGT"
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True)
        assert set(tokens.keys()) == {"input_ids", "attention_mask", "position_ids", "het_values"}

    def test_attention_mask_all_ones(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 10
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True)
        assert all(m == 1 for m in tokens["attention_mask"])

    def test_attention_mask_length_matches_input_ids(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 15
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True)
        assert len(tokens["attention_mask"]) == len(tokens["input_ids"])

    def test_position_ids_range(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 25  # 100 bp
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True)
        L = len(seq)
        assert all(0 <= p < L for p in tokens["position_ids"])

    def test_position_ids_stride1_circular(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 25  # 100 bp
        tokens = tokenize_sequence(seq, vocabulary=vocab, stride=1, circular=True)
        # Position IDs must be 0, 1, 2, ..., 99 in order
        assert tokens["position_ids"] == list(range(len(seq)))

    def test_max_seq_len_truncation(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 200  # 800 bp
        tokens = tokenize_sequence(seq, vocabulary=vocab, max_seq_len=64, circular=True)
        assert len(tokens["input_ids"]) == 64

    def test_n_maps_to_unk(self, vocab: KmerVocabulary) -> None:
        seq = "NNNNNN" + "ACGT" * 10
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=False)
        assert tokens["input_ids"][0] == UNK_TOKEN_ID

    def test_case_insensitive(self, vocab: KmerVocabulary) -> None:
        seq_upper = "ACGTAC" * 5
        seq_lower = "acgtac" * 5
        t_upper = tokenize_sequence(seq_upper, vocabulary=vocab, circular=True)
        t_lower = tokenize_sequence(seq_lower, vocabulary=vocab, circular=True)
        assert t_upper["input_ids"] == t_lower["input_ids"]

    def test_het_values_default_zeros(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 10
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True)
        assert all(v == 0.0 for v in tokens["het_values"])

    def test_het_values_provided(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 10  # 40 bp
        het = np.linspace(0.0, 1.0, len(seq))
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True, het_levels=het)
        assert len(tokens["het_values"]) == len(seq)
        assert all(0.0 <= v <= 1.0 for v in tokens["het_values"])

    def test_het_values_length_matches(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 10
        het = np.zeros(len(seq))
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True, het_levels=het)
        assert len(tokens["het_values"]) == len(tokens["input_ids"])

    def test_all_ids_valid(self, vocab: KmerVocabulary) -> None:
        seq = "ACGT" * 25
        tokens = tokenize_sequence(seq, vocabulary=vocab, circular=True)
        for tid in tokens["input_ids"]:
            assert 0 <= tid < len(vocab)
