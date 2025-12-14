#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_acc_vs_k.py

Load models/<ds>_test_preds_from_<ds>_train_k{K}.npz files,
compute accuracy for each K, and plot an accuracy-vs-K line chart.

Optional:
- Also plot DTree accuracy and Majority baseline.

Usage:
  python plot_acc_vs_k.py --dataset mnist
  python plot_acc_vs_k.py --dataset fashion --ks 2,4,8,16,20,32,64 --with-dtree --with-majority
"""

import os
import re
import glob
import argparse
from collections import Counter
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

FMT_LIST  = ["Dense", "CSR", "BCSR", "DIA", "COO", "LIL"]
FLOW_LIST = ["IP", "OP"]

def pack_label(fmt: str, flow: str) -> int:
    return FMT_LIST.index(fmt) * len(FLOW_LIST) + FLOW_LIST.index(flow)

def load_preds(npz_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    ids = list(z["ids"])
    y_pred = z["y_pred"].astype(np.int64)
    y_true = z["y_true"] if "y_true" in z else np.array([], dtype=np.int64)
    z.close()
    return ids, y_pred, y_true

def reconstruct_y_true(ids: List[str], best_npz: str) -> np.ndarray | None:
    if not os.path.exists(best_npz):
        return None
    z = np.load(best_npz, allow_pickle=True)
    ids_bc = list(z["ids"])
    fmts   = list(z["best_fmt"])
    flows  = list(z["best_flow"])
    z.close()
    id2lbl = {ids_bc[i]: pack_label(fmts[i], flows[i]) for i in range(len(ids_bc))}
    y_true = np.array([id2lbl.get(i, -1) for i in ids], dtype=np.int64)
    if (y_true < 0).any():
        return None
    return y_true

def majority_label(y_true: np.ndarray) -> int:
    from collections import Counter
    cnt = Counter(y_true.tolist())
    return int(cnt.most_common(1)[0][0])

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float((y_pred == y_true).mean())

def main():
    ap = argparse.ArgumentParser("Plot accuracy vs K for centers predictions")
    ap.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--ks", default=None,
                    help="comma-separated K list (e.g., 2,4,8,16,20,32,64). "
                         "If omitted, will auto-discover by glob.")
    ap.add_argument("--with-dtree", action="store_true", help="also plot DTree accuracy")
    ap.add_argument("--with-majority", action="store_true", help="also plot Majority baseline")
    args = ap.parse_args()

    ds = args.dataset.lower()
    models_dir = args.models_dir
    cache_dir = args.cache_dir

    # discover files
    if args.ks is None:
        pattern = os.path.join(models_dir, f"{ds}_test_preds_from_{ds}_train_k*.npz")
        paths = sorted(glob.glob(pattern))
        items = []
        for p in paths:
            m = re.search(r"_k(\d+)\.npz$", p)
            if m:
                items.append((int(m.group(1)), p))
        items.sort(key=lambda x: x[0])
    else:
        Ks = [int(x) for x in args.ks.split(",")]
        items = []
        for k in Ks:
            p = os.path.join(models_dir, f"{ds}_test_preds_from_{ds}_train_k{k}.npz")
            if os.path.exists(p):
                items.append((k, p))
        items.sort(key=lambda x: x[0])

    if not items:
        raise SystemExit("No center prediction files found.")

    # reference & y_true
    ref_k, ref_path = items[0]
    ids_ref, _, y_true_ref = load_preds(ref_path)
    best_npz = os.path.join(cache_dir, f"best_choices_{ds}_test_sparse.npz")
    if y_true_ref.size == 0 or (y_true_ref < 0).any():
        y_true = reconstruct_y_true(ids_ref, best_npz)
        if y_true is None:
            raise RuntimeError("Cannot obtain y_true for test set.")
    else:
        y_true = y_true_ref

    # gather accuracies
    K_list, acc_list = [], []
    for k, path in items:
        ids, y_pred, _ = load_preds(path)
        if ids != ids_ref:
            # align by id map
            map_pred = {iid: y for iid, y in zip(ids, y_pred)}
            y_pred = np.array([map_pred[i] for i in ids_ref], dtype=np.int64)
        acc = accuracy(y_pred, y_true)
        K_list.append(k)
        acc_list.append(acc)

    # optional dtree
    acc_dtree = None
    if args.with_dtree:
        p_dt = os.path.join(models_dir, f"{ds}_test_preds_from_dtree.npz")
        if os.path.exists(p_dt):
            ids_dt, y_dt, y_true_dt = load_preds(p_dt)
            if ids_dt != ids_ref:
                map_dt = {iid: y for iid, y in zip(ids_dt, y_dt)}
                y_dt = np.array([map_dt[i] for i in ids_ref], dtype=np.int64)
            if y_true_dt.size == 0 or (y_true_dt < 0).any():
                y_true_dt = y_true
            acc_dtree = accuracy(y_dt, y_true_dt)

    # optional majority
    acc_major = None
    if args.with_majority:
        maj = majority_label(y_true)
        y_maj = np.full_like(y_true, maj)
        acc_major = accuracy(y_maj, y_true)

    # ---- plot ----
    plt.figure(figsize=(9.2, 5.6))

    # centers line
    plt.plot(K_list, acc_list, marker="o", linewidth=2.2, markersize=7,
             color="#4C78A8", label="Centers")

    # 在每個藍點上標「K=..」與 accuracy
    for x, y in zip(K_list, acc_list):
        plt.text(x, y + 0.011, f"K={x}\n{y*100:.2f}%",
                 ha="center", va="bottom", fontsize=8)

    # dtree / majority 參考線
    if acc_dtree is not None:
        plt.axhline(acc_dtree, color="#F58518", linestyle="--", linewidth=1.8,
                    label=f"DTree ({acc_dtree*100:.2f}%)")
    if acc_major is not None:
        plt.axhline(acc_major, color="#54A24B", linestyle=":", linewidth=1.8,
                    label=f"Majority ({acc_major*100:.2f}%)")

    plt.xlabel("K (number of centers)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"{ds.upper()} test accuracy vs K", fontsize=20, pad=40)

    # 預留標籤空間
    ymin = max(0.0, min(acc_list) - 0.04)
    ymax = min(1.0, max(acc_list) + 0.07)
    plt.ylim(ymin, ymax)

    # 格線 + legend 放上方、出框
    plt.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=12)

    # 為上方 legend + title 空出頂部
    plt.tight_layout(rect=[0, 0, 1, 0.86])

    out_png = os.path.join(models_dir, f"{ds}_test_acc_vs_k.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_png}")

if __name__ == "__main__":
    main()
