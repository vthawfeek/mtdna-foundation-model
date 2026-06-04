"""
Push mtDNA-FM models and LoRA adapters to HuggingFace Hub.

Usage:
    uv run python mtdna_fm/scripts/push_to_hub.py

This script:
  1. Pushes the base model (phase1_v1) to vthawfeek/mtdna-foundation-model
  2. Pushes the trained haplogroup LoRA adapter to vthawfeek/mtdna-fm-haplogroup
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from mtdna_fm.model.model import MtDNAForMaskedModeling

REPO_BASE = "vthawfeek/mtdna-foundation-model"
REPO_HAPLOGROUP = "vthawfeek/mtdna-fm-haplogroup"

PHASE1_DIR = Path("models/phase1_v1")
HAPLOGROUP_DIR = Path("models/finetune_haplogroup_paper")

HAPLOGROUP_README = """---
license: apache-2.0
base_model: vthawfeek/mtdna-foundation-model
tags:
- biology
- genomics
- mitochondrial-dna
- peft
- lora
- haplogroup-classification
pipeline_tag: text-classification
---

# mtDNA-FM Haplogroup Adapter (LoRA r=8)

LoRA adapter for haplogroup classification (26 major haplogroups) on top of
[vthawfeek/mtdna-foundation-model](https://huggingface.co/vthawfeek/mtdna-foundation-model).

## Usage

```python
from mtdna_fm.model.model import MtDNAForHaplogroupClassification, MtDNAModel
from peft import PeftModel

base = MtDNAModel.from_pretrained("vthawfeek/mtdna-foundation-model")
model = MtDNAForHaplogroupClassification(base, num_labels=26)
model = PeftModel.from_pretrained(model, "vthawfeek/mtdna-fm-haplogroup")
model.eval()
```

## LoRA Configuration

- r = 8, lora_alpha = 16
- target_modules: query, key, value, dense
- lora_dropout = 0.1

## Task

26-class haplogroup classification: A, B, C, D, E, F, G, H, HV, I, J, K,
L0, L1, L2, L3, L4, L5, M, N, R, T, U, V, W, X.

Input: full 16,569-bp mtDNA genome sequence. Mean-pooled CLS embeddings across
overlapping windows (size=512, stride=256) fed into a Linear(256, 26) head.

## Limitations

HmtDB training data has a European population bias (haplogroup H is overrepresented).
Performance on underrepresented African L sub-haplogroups may be lower.
"""


def push_base_model(api: HfApi) -> None:
    """Push base model + tokenizer + model card to Hub."""
    print(f"\n=== Pushing base model to {REPO_BASE} ===")

    api.create_repo(repo_id=REPO_BASE, repo_type="model", exist_ok=True)
    print(f"  Repository {REPO_BASE} ready")

    for fname in ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.json", "README.md"]:
        fpath = PHASE1_DIR / fname
        if not fpath.exists():
            print(f"  Skipping {fname} (not found)")
            continue
        print(f"  Uploading {fname}...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=REPO_BASE,
            repo_type="model",
        )
        print(f"  ✓ {fname}")


def _push_adapter(
    api: HfApi,
    source_dir: Path,
    repo_id: str,
    readme_content: str,
) -> None:
    """Upload a trained LoRA adapter directory to the Hub, patching the config path."""
    print(f"\n=== Pushing adapter to {repo_id} ===")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Prepare a temp dir with patched adapter_config.json and proper README
    tmp_dir = Path(f"models/_hub_tmp_{repo_id.split('/')[-1]}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Copy adapter weights unchanged
        shutil.copy(source_dir / "adapter_model.safetensors", tmp_dir / "adapter_model.safetensors")

        # Patch config to use Hub model path instead of local path
        config = json.loads((source_dir / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = REPO_BASE
        (tmp_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))

        # Write proper README
        (tmp_dir / "README.md").write_text(readme_content)

        for fpath in tmp_dir.iterdir():
            if fpath.is_file():
                print(f"  Uploading {fpath.name}...")
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fpath.name,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"  ✓ {fpath.name}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def verify_hub_load() -> None:
    """Smoke-test that the Hub base model loads correctly."""
    print(f"\n=== Verifying Hub load for {REPO_BASE} ===")
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(REPO_BASE)
        print(f"  AutoConfig: model_type={cfg.model_type}, hidden_size={cfg.hidden_size}")
    except Exception as e:
        print(f"  AutoConfig failed (may need mtdna_fm registered): {e}")

    model = MtDNAForMaskedModeling.from_pretrained(REPO_BASE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MtDNAForMaskedModeling: {n_params:,} parameters")
    print("  ✓ Hub load verified")


def main() -> None:
    api = HfApi()
    user_info = api.whoami()
    print(f"Authenticated as: {user_info['name']}")

    push_base_model(api)
    _push_adapter(api, HAPLOGROUP_DIR, REPO_HAPLOGROUP, HAPLOGROUP_README)

    verify_hub_load()

    print("\n✓ HuggingFace Hub push complete!")
    print(f"  Base model:   https://huggingface.co/{REPO_BASE}")
    print(f"  Haplogroup:   https://huggingface.co/{REPO_HAPLOGROUP}")


if __name__ == "__main__":
    main()
