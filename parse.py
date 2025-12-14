#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse.py
- 將 data/cache/<prefix>.npz 內的 images(uint8) 依 thr 二值化為 masks(bool)
- 產生 data/cache/<prefix>_sparse.npz
- 嚴格對齊 IDs：<file_stem>:images[i]
- 不做子抽樣；不做額外過濾（只依 threshold）
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np

CACHE_DIR = Path("data/cache")

def binarize_images_uint8(X: np.ndarray, thr: float) -> np.ndarray:
    """
    X: uint8 [N,H,W] or [N,H,W,1]
    thr: 0~1 之間的比例（相對 255）
    return: bool masks [N,H,W]
    """
    if X.ndim == 4 and X.shape[-1] == 1:
        X = X[..., 0]
    if X.dtype != np.uint8 or X.ndim != 3:
        raise ValueError(f"images must be uint8 [N,H,W] (or [N,H,W,1]), got {X.dtype}, {X.shape}")
    t = int(round(thr * 255.0))
    return (X > t)

def process_one_npz(npz_path: Path, thr: float):
    print(f"[parse] {npz_path}")
    z = np.load(npz_path, allow_pickle=False)
    if "images" not in z:
        z.close()
        raise ValueError(f"{npz_path.name} must contain key 'images' (uint8 [N,H,W])")
    X = z["images"]
    labels = z["labels"] if "labels" in z else None
    z.close()

    masks = binarize_images_uint8(X, thr)
    N, H, W = masks.shape
    ids = np.array([f"{npz_path.stem}:images[{i}]" for i in range(N)], dtype=object)

    out_path = npz_path.with_name(f"{npz_path.stem}_sparse.npz")
    np.savez_compressed(
        out_path,
        ids=ids,
        masks=masks.astype(np.bool_),
        labels=(labels if labels is not None else np.array([], dtype=np.int64)),
        source=np.array(str(npz_path), dtype=object),
        threshold=np.float32(thr)
    )
    print(f"  -> wrote {out_path.name}  N={N}  HxW={H}x{W}  thr={thr}")

def main():
    ap = argparse.ArgumentParser("Binarize images to sparse masks")
    ap.add_argument("--prefix", type=str, required=True,
                    help="file stem in data/cache, e.g. mnist_train / fashion_test")
    ap.add_argument("--thr", type=float, default=0.20,
                    help="threshold (0~1) w.r.t. 255 for binarization")
    args = ap.parse_args()

    npz_path = CACHE_DIR / f"{args.prefix}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    process_one_npz(npz_path, args.thr)

if __name__ == "__main__":
    main()
