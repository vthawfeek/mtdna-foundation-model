"""
K-mer vocabulary for DNA sequence tokenization.

Deterministic: build(k=6) always produces the same token-to-index mapping
regardless of platform or Python version. Special tokens occupy indices 0-5;
k-mer tokens are sorted lexicographically starting at index 6.
"""

import itertools
import json
from pathlib import Path

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[MASK]", "[UNK]", "[SEP]", "[HET]"]

PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 1
MASK_TOKEN_ID = 2
UNK_TOKEN_ID = 3
SEP_TOKEN_ID = 4
HET_TOKEN_ID = 5


class KmerVocabulary:
    """Deterministic k-mer vocabulary for any ACGT-alphabet genome."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self._token_to_id = token_to_id
        self._id_to_token = {v: k for k, v in token_to_id.items()}

    # ── Convenience accessors ──────────────────────────────────────────────────
    @property
    def pad_token_id(self) -> int:
        return PAD_TOKEN_ID

    @property
    def cls_token_id(self) -> int:
        return CLS_TOKEN_ID

    @property
    def mask_token_id(self) -> int:
        return MASK_TOKEN_ID

    @property
    def unk_token_id(self) -> int:
        return UNK_TOKEN_ID

    @property
    def sep_token_id(self) -> int:
        return SEP_TOKEN_ID

    @property
    def het_token_id(self) -> int:
        return HET_TOKEN_ID

    @property
    def n_special(self) -> int:
        """Number of special tokens (real k-mer IDs start at this index)."""
        return len(SPECIAL_TOKENS)

    @classmethod
    def build(cls, k: int = 6) -> "KmerVocabulary":
        """Build vocabulary of all 4^k k-mers plus 6 special tokens."""
        token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        idx = len(SPECIAL_TOKENS)
        for bases in sorted(itertools.product("ACGT", repeat=k)):
            token_to_id["".join(bases)] = idx
            idx += 1
        return cls(token_to_id)

    def encode(self, token: str) -> int:
        """Return token ID; unknown tokens map to UNK."""
        return self._token_to_id.get(token, UNK_TOKEN_ID)

    def decode(self, token_id: int) -> str:
        """Return token string for a given ID."""
        return self._id_to_token.get(token_id, "[UNK]")

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def save_pretrained(self, path: str | Path) -> None:
        """Save vocabulary to directory (HuggingFace-style convention)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "vocab.json", "w") as f:
            json.dump(self._token_to_id, f)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "KmerVocabulary":
        """Load vocabulary from a directory produced by save_pretrained."""
        path = Path(path)
        with open(path / "vocab.json") as f:
            token_to_id = json.load(f)
        return cls(token_to_id)
