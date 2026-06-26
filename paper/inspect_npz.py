import numpy as np

for fname in ["app_reference.npz", "app_patho_reference.npz", "reports/zeroshot_patho_embeddings.npz"]:
    d = np.load(fname, allow_pickle=True)
    print(f"--- {fname} ---")
    for k in d.files:
        v = d[k]
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        if v.size > 0 and v.dtype.kind not in ("U", "S", "O"):
            print(f"    min={v.min():.4f}, max={v.max():.4f}")
        elif v.size > 0:
            sample = v.flat[:3]
            print(f"    sample={list(sample)}")
    print()
