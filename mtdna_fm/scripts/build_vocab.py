"""Build and save the k-mer vocabulary to disk.

DVC build_vocabulary stage entrypoint:
    uv run python mtdna_fm/scripts/build_vocab.py <output_dir>

The vocabulary is deterministic (all 4^k k-mers + 6 special tokens), so
this stage has no data deps — only the source file dep in dvc.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(output_dir: str = "data/processed/vocabulary", k: int = 6) -> None:
    from mtdna_fm.tokenizer.vocabulary import KmerVocabulary

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = KmerVocabulary.build(k=k)
    vocab.save_pretrained(str(out))
    n = len(vocab)
    print(f"[build-vocab] Saved {n} tokens ({4**k} {k}-mers + 6 special) to {out}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/processed/vocabulary"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    main(out, k)
