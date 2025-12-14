#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize centers (basis) saved by centers_decider_latency.py

Usage examples:
  # just draw the basis grid
  python viz_centers.py --trained-prefix mnist_train --k 32

  # also compute & show cluster sizes using <dataset>_<split>_sparse.npz
  python viz_centers.py --trained-prefix fashion_train --k 128 \
      --dataset fashion --split train

Options:
  --sort nnz|count|none   : order tiles by nnz (dense first), by cluster size, or leave as-is
  --cols 8                : number of columns in the grid
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

CACHE_DIR  = Path("data/cache")
MODELS_DIR = Path("models")

def _unpack(lbl, fmt_list, flow_list):
    i, j = divmod(int(lbl), len(flow_list))
    return f"{fmt_list[i]}_{flow_list[j]}"

def _nearest_by_jaccard(masks, centers, batch=4096):
    """
    Assign masks -> nearest center by Jaccard similarity (fast & memory-safe).
    masks:   [N,H,W] bool
    centers: [K,H,W] bool
    return labels[N] in [0,K)
    """
    N, H, W = masks.shape
    K = centers.shape[0]
    X = masks.reshape(N, -1).astype(np.uint8)
    C = centers.reshape(K, -1).astype(np.uint8)
    sum_x = X.sum(axis=1, dtype=np.int32)
    sum_c = C.sum(axis=1, dtype=np.int32)
    labels = np.empty(N, dtype=np.int32)
    for s in tqdm(range(0, N, batch), desc="assign (Jaccard)", leave=False):
        e = min(N, s + batch)
        inter = X[s:e] @ C.T                      # [B,K]
        union = sum_x[s:e, None] + sum_c[None, :] - inter
        sim = np.where(union > 0, inter / np.maximum(1, union), 1.0)
        labels[s:e] = np.argmax(sim, axis=1)
    return labels

def visualize(centers_pkl: Path, counts=None, sort="none", cols=8, dpi=200, cmap="gray"):
    b = joblib.load(centers_pkl)
    centers   = b["centers"].astype(bool)  # [K,H,W]
    labels    = b["center_labels"]         # packed ints
    fmt_list  = list(b["fmt_list"])
    flow_list = list(b["flow_list"])
    K, H, W = centers.shape

    # sorting
    order = np.arange(K)
    if sort == "nnz":
        nnz = centers.reshape(K, -1).sum(axis=1)
        order = np.argsort(-nnz)  # dense first
    elif sort == "count" and counts is not None:
        order = np.argsort(-counts)

    centers = centers[order]
    labels  = labels[order]
    counts  = (counts[order] if counts is not None else None)

    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.0, rows*2.0))
    axes = np.atleast_2d(axes)

    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.axis("off")
        if k >= K: continue
        ax.imshow(centers[k], cmap=cmap, interpolation="nearest")
        title = _unpack(int(labels[k]), fmt_list, flow_list)
        if counts is not None:
            title += f"\ncount={int(counts[k])}"
        ax.set_title(title, fontsize=9, pad=2)

    plt.tight_layout(pad=0.2)
    out_png = centers_pkl.with_suffix(".png")
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] saved -> {out_png}")

def main():
    ap = argparse.ArgumentParser("Visualize basis (centers)")
    ap.add_argument("--trained-prefix", required=True, help="e.g., mnist_train, fashion_train")
    ap.add_argument("--k", type=int, required=True, help="K used during fit")
    ap.add_argument("--dataset", choices=["mnist","fashion"], default=None,
                    help="if given, compute cluster sizes on this split")
    ap.add_argument("--split", choices=["train","test"], default="train")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--sort", choices=["none","nnz","count"], default="none")
    args = ap.parse_args()

    centers_pkl = MODELS_DIR / f"{args.trained_prefix}_centers_k{args.k}.pkl"
    if not centers_pkl.exists():
        raise FileNotFoundError(centers_pkl)

    counts = None
    if args.dataset is not None:
        prefix = f"{args.dataset}_{args.split}"
        npz_path = CACHE_DIR / f"{prefix}_sparse.npz"
        z = np.load(npz_path, allow_pickle=True)
        masks = z["masks"].astype(bool)
        z.close()

        b = joblib.load(centers_pkl)
        centers = b["centers"].astype(bool)
        print(f"[INFO] tallying cluster sizes on {prefix} (N={len(masks)}) ...")
        assigned = _nearest_by_jaccard(masks, centers)
        counts = np.bincount(assigned, minlength=centers.shape[0])

    visualize(centers_pkl, counts=counts, sort=args.sort, cols=args.cols, dpi=args.dpi)

if __name__ == "__main__":
    main()
