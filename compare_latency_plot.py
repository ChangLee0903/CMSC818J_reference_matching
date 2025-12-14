#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_latency_relative.py

Plot relative latency (vs. Oracle) for:
  - DTree predictions
  - Centers predictions
  - Majority baseline (single most frequent label)
Dataset: mnist / fashion

Inputs (expected paths):
  models/<ds>_test_preds_from_dtree.npz
  models/<ds>_test_preds_from_<ds>_train_k{K}.npz
  data/cache/<ds>_test_sparse.npz                  # masks + ids
  data/cache/best_choices_<ds>_test_sparse.npz     # to reconstruct y_true when needed

Output:
  models/<ds>_test_latency_relative_k{K}.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Tuple, List, Dict

# ---- estimator ----
from simulation import estimate_spmv  # 確保同目錄或可 import

FMT_LIST  = ["Dense", "CSR", "BCSR", "DIA", "COO", "LIL"]
FLOW_LIST = ["IP", "OP"]

def pack_label(fmt: str, flow: str) -> int:
    return FMT_LIST.index(fmt) * len(FLOW_LIST) + FLOW_LIST.index(flow)

def unpack_label(lbl: int) -> Tuple[str, str]:
    i, j = divmod(int(lbl), len(FLOW_LIST))
    return FMT_LIST[i], FLOW_LIST[j]

def load_preds(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    ids = list(z["ids"])
    y_pred = z["y_pred"].astype(np.int64)
    y_true = z["y_true"] if "y_true" in z else np.array([], dtype=np.int64)
    z.close()
    return ids, y_pred, y_true

def load_sparse_masks(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    masks = z["masks"].astype(bool)
    ids = list(z["ids"])
    z.close()
    return masks, ids

def reconstruct_y_true_from_best(ids: List[str], best_path: str):
    if not os.path.exists(best_path):
        return None
    z = np.load(best_path, allow_pickle=True)
    ids_bc = list(z["ids"])
    fmts   = list(z["best_fmt"])
    flows  = list(z["best_flow"])
    z.close()
    id2lbl = {ids_bc[i]: pack_label(fmts[i], flows[i]) for i in range(len(ids_bc))}
    y = np.array([id2lbl.get(i, -1) for i in ids], dtype=np.int64)
    if (y < 0).any():
        return None
    return y

def majority_label(y_true: np.ndarray) -> int:
    cnt = Counter(y_true.tolist())
    return int(cnt.most_common(1)[0][0])

def avg_latency_for_preds(masks: np.ndarray, y_lbls: np.ndarray) -> float:
    N = masks.shape[0]
    total = 0
    for i in range(N):
        fmt, flow = unpack_label(int(y_lbls[i]))
        s = estimate_spmv(masks[i].astype(int), fmt, dataflow=flow)
        total += int(s["latency"])
    return total / float(N)

def avg_latency_for_single_label(masks: np.ndarray, lbl: int) -> float:
    fmt, flow = unpack_label(int(lbl))
    total = 0
    for i in range(masks.shape[0]):
        s = estimate_spmv(masks[i].astype(int), fmt, dataflow=flow)
        total += int(s["latency"])
    return total / float(masks.shape[0])

def main():
    ap = argparse.ArgumentParser("Relative latency bars vs Oracle")
    ap.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    ap.add_argument("--centers-k", type=int, default=32)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--ymin", type=float, default=None, help="y-axis lower bound (e.g., 0.85)")
    ap.add_argument("--ymax", type=float, default=None, help="y-axis upper bound (e.g., 1.15)")
    args = ap.parse_args()

    ds = args.dataset.lower()
    k = args.centers_k
    models_dir = args.models_dir
    cache_dir = args.cache_dir

    # files
    pred_dtree = os.path.join(models_dir, f"{ds}_test_preds_from_dtree.npz")
    pred_cent  = os.path.join(models_dir, f"{ds}_test_preds_from_{ds}_train_k{k}.npz")
    sparse_npz = os.path.join(cache_dir, f"{ds}_test_sparse.npz")
    best_npz   = os.path.join(cache_dir, f"best_choices_{ds}_test_sparse.npz")

    # load predictions
    ids_dt, y_dt, y_true_dt = load_preds(pred_dtree)
    ids_ce, y_ce, y_true_ce = load_preds(pred_cent)

    # unify ground-truth labels
    y_true = None
    ids_ref = None
    if y_true_dt.size > 0 and (y_true_dt >= 0).all():
        y_true, ids_ref = y_true_dt, ids_dt
    elif y_true_ce.size > 0 and (y_true_ce >= 0).all():
        y_true, ids_ref = y_true_ce, ids_ce
    else:
        ids_ref = ids_dt
        y_try = reconstruct_y_true_from_best(ids_ref, best_npz)
        if y_try is None:
            raise RuntimeError("Cannot obtain y_true for test set.")
        y_true = y_try

    # align second prediction to ids_ref if needed
    if ids_ce != ids_ref:
        map_ce = {iid: y for iid, y in zip(ids_ce, y_ce)}
        y_ce = np.array([map_ce[i] for i in ids_ref], dtype=np.int64)
    if ids_dt != ids_ref:
        map_dt = {iid: y for iid, y in zip(ids_dt, y_dt)}
        y_dt = np.array([map_dt[i] for i in ids_ref], dtype=np.int64)

    # load masks & align order
    masks, ids_masks = load_sparse_masks(sparse_npz)
    if ids_masks != ids_ref:
        idx_map = {iid: i for i, iid in enumerate(ids_masks)}
        order = [idx_map[iid] for iid in ids_ref]
        masks = masks[np.array(order, dtype=np.int64)]

    # compute average latencies
    print(f"[INFO] N={len(ids_ref)} — computing average latencies ...")
    lat_oracle = avg_latency_for_preds(masks, y_true)
    lat_dtree  = avg_latency_for_preds(masks, y_dt)
    lat_center = avg_latency_for_preds(masks, y_ce)
    maj_lbl    = majority_label(y_true)
    lat_major  = avg_latency_for_single_label(masks, maj_lbl)

    # relative to oracle
    rel_dtree  = lat_dtree  / lat_oracle
    rel_center = lat_center / lat_oracle
    rel_major  = lat_major  / lat_oracle

    labels = ["DTree", "Centers", "Majority"]
    values = [rel_dtree, rel_center, rel_major]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    plt.figure(figsize=(8.2, 5.2))
    xs = np.arange(len(labels))
    bars = plt.bar(xs, values, color=colors, edgecolor="#333333", linewidth=1.0)

    for x, v in zip(xs, values):
        plt.text(x, v + 0.01, f"{v:.3f}×", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#222222")

    plt.axhline(1.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7, label="Oracle = 1.0×")
    plt.xticks(xs, labels, fontsize=12)

    # ---- y-axis range (提升對比) ----
    if args.ymin is not None or args.ymax is not None:
        ymin = 0.0 if args.ymin is None else args.ymin
        ymax = max(1.05, max(values) + 0.06) if args.ymax is None else args.ymax
    else:
        # 自動提高起點：貼著最小值往下留一點點空間
        vmin, vmax = min(values), max(values)
        ymin = max(0.70, vmin - 0.05)          # 例如 0.70～0.95 之間，視資料而定
        ymax = max(1.05, vmax + 0.06)
    plt.ylim(ymin, ymax)

    yt = plt.gca().get_yticks()
    plt.yticks(yt, [f"{t:.2f}×" for t in yt], fontsize=10)
    plt.gca().yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    plt.gca().set_axisbelow(True)

    plt.title(f"{ds.upper()} relative latency vs Oracle (K={k})", fontsize=14, pad=10)
    plt.ylabel("Relative latency (lower is better)", fontsize=12)
    plt.legend(loc="upper right", frameon=False, fontsize=10)

    plt.tight_layout()
    out_png = os.path.join(models_dir, f"{ds}_test_latency_relative_k{k}.png")
    os.makedirs(models_dir, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.show()

    print("\n[Summary] Average latency (cycles) and relative to Oracle")
    print(f"  Oracle : {lat_oracle:.2f}  (1.000×)")
    print(f"  DTree  : {lat_dtree:.2f}  ({rel_dtree:.3f}×)")
    print(f"  Centers: {lat_center:.2f}  ({rel_center:.3f}×)")
    print(f"  Majority: {lat_major:.2f} ({rel_major:.3f}×)")
    print(f"Saved plot: {out_png}")

if __name__ == "__main__":
    main()
