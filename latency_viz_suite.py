#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latency_viz_suite.py

Produce rich latency comparisons:
  1) Mean & geometric-mean latency bars (Oracle, DTree, Centers, Majority)
  2) CDF of per-sample latency ratios vs Oracle
  3) Histogram of ratios
  4) Scatter: Oracle vs methods (log-scale optional)
  5) Per-true-label (fmt_flow) ratio bars (geometric mean)

Inputs expected:
  models/<ds>_test_preds_from_dtree.npz
  models/<ds>_test_preds_from_<ds>_train_k{K}.npz
  data/cache/<ds>_test_sparse.npz
  data/cache/best_choices_<ds>_test_sparse.npz

Outputs:
  models/<ds>_latency_suite_k{K}_*.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import Tuple, List, Dict

# ---- estimator (make sure it's importable) ----
from simulation import estimate_spmv

FMT_LIST  = ["Dense", "CSR", "BCSR", "DIA", "COO", "LIL"]
FLOW_LIST = ["IP", "OP"]

def pack_label(fmt: str, flow: str) -> int:
    return FMT_LIST.index(fmt) * len(FLOW_LIST) + FLOW_LIST.index(flow)

def unpack_label(lbl: int) -> Tuple[str, str]:
    i, j = divmod(int(lbl), len(FLOW_LIST))
    return FMT_LIST[i], FLOW_LIST[j]

def label_name(lbl: int) -> str:
    f, d = unpack_label(int(lbl))
    return f"{f}_{d}"

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
    ids_bc = list(z["ids"]); fmts = list(z["best_fmt"]); flows = list(z["best_flow"])
    z.close()
    id2lbl = {ids_bc[i]: pack_label(fmts[i], flows[i]) for i in range(len(ids_bc))}
    y = np.array([id2lbl.get(i, -1) for i in ids], dtype=np.int64)
    if (y < 0).any():  # missing GT → give up
        return None
    return y

def align_to(ids_src: List[str], vals_src: np.ndarray, ids_ref: List[str]) -> np.ndarray:
    if ids_src == ids_ref:
        return vals_src
    m = {i:v for i, v in zip(ids_src, vals_src)}
    return np.array([m[i] for i in ids_ref], dtype=vals_src.dtype)

def majority_label(y_true: np.ndarray) -> int:
    cnt = Counter(y_true.tolist())
    return int(cnt.most_common(1)[0][0])

def per_sample_latency(masks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    N = masks.shape[0]
    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        f, d = unpack_label(int(labels[i]))
        s = estimate_spmv(masks[i].astype(int), f, dataflow=d)
        out[i] = int(s["latency"])
    return out

def geomean(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.clip(x, eps, None)
    return float(np.exp(np.mean(np.log(x))))

def main():
    ap = argparse.ArgumentParser("Latency visualization suite")
    ap.add_argument("--dataset", choices=["mnist","fashion"], required=True)
    ap.add_argument("--centers-k", type=int, default=32)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--logy", action="store_true", help="use log-scale on Y when helpful")
    args = ap.parse_args()

    ds = args.dataset.lower()
    k = args.centers_k
    models_dir = args.models_dir
    cache_dir = args.cache_dir

    # paths
    pred_dtree = os.path.join(models_dir, f"{ds}_test_preds_from_dtree.npz")
    pred_cent  = os.path.join(models_dir, f"{ds}_test_preds_from_{ds}_train_k{k}.npz")
    sparse_npz = os.path.join(cache_dir, f"{ds}_test_sparse.npz")
    best_npz   = os.path.join(cache_dir, f"best_choices_{ds}_test_sparse.npz")

    # load predictions
    ids_dt, y_dt, y_true_dt = load_preds(pred_dtree)
    ids_ce, y_ce, y_true_ce = load_preds(pred_cent)

    # choose reference ids & y_true
    if y_true_dt.size > 0 and (y_true_dt >= 0).all():
        ids_ref, y_true = ids_dt, y_true_dt
    elif y_true_ce.size > 0 and (y_true_ce >= 0).all():
        ids_ref, y_true = ids_ce, y_true_ce
    else:
        ids_ref = ids_dt
        y_true = reconstruct_y_true_from_best(ids_ref, best_npz)
        if y_true is None:
            raise RuntimeError("Cannot obtain y_true for the test set.")

    # align the other preds
    y_ce = align_to(ids_ce, y_ce, ids_ref)
    y_dt = align_to(ids_dt, y_dt, ids_ref)

    # load masks & align order
    masks, ids_masks = load_sparse_masks(sparse_npz)
    if ids_masks != ids_ref:
        idx_map = {iid:i for i, iid in enumerate(ids_masks)}
        order = np.array([idx_map[i] for i in ids_ref], dtype=np.int64)
        masks = masks[order]

    # compute per-sample latencies
    print(f"[INFO] N={len(ids_ref)}  computing per-sample latencies ...")
    lat_oracle = per_sample_latency(masks, y_true)
    lat_dtree  = per_sample_latency(masks, y_dt)
    lat_center = per_sample_latency(masks, y_ce)
    maj_lbl    = majority_label(y_true)
    lat_major  = per_sample_latency(masks, np.full_like(y_true, maj_lbl))

    # 1) mean & geometric mean bars (absolute cycles)
    fig1 = plt.figure(figsize=(8.4, 5.2))
    names  = ["Oracle", "DTree", "Centers", "Majority"]
    means  = [lat_oracle.mean(), lat_dtree.mean(), lat_center.mean(), lat_major.mean()]
    gmeans = [geomean(lat_oracle), geomean(lat_dtree), geomean(lat_center), geomean(lat_major)]
    xs = np.arange(len(names))
    w  = 0.36
    b1 = plt.bar(xs - w/2, means, width=w, label="Mean", color="#4C78A8")
    b2 = plt.bar(xs + w/2, gmeans, width=w, label="Geometric mean", color="#F58518")
    for rect in b1+b2:
        h = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2, h*1.01, f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    if args.logy:
        plt.yscale("log")
    plt.ylabel("Latency (cycles)")
    plt.title(f"{ds.upper()} latency — mean & geometric mean (K={k})")
    plt.xticks(xs, names)
    plt.gca().yaxis.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    out1 = os.path.join(models_dir, f"{ds}_latency_means_k{k}.png")
    plt.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    print(f"[SAVE] {out1}")

    # ratios vs Oracle
    r_dt  = lat_dtree  / lat_oracle
    r_ce  = lat_center / lat_oracle
    r_mj  = lat_major  / lat_oracle

    # 2) CDF of ratios
    def cdf_vals(x):
        x = np.sort(x)
        y = np.linspace(0, 1, len(x), endpoint=True)
        return x, y
    x1,y1 = cdf_vals(r_dt); x2,y2 = cdf_vals(r_ce); x3,y3 = cdf_vals(r_mj)

    fig2 = plt.figure(figsize=(7.6, 5.0))
    plt.plot(x1, y1, label=f"DTree (g-mean {geomean(r_dt):.3f}×)", lw=2)
    plt.plot(x2, y2, label=f"Centers (g-mean {geomean(r_ce):.3f}×)", lw=2)
    plt.plot(x3, y3, label=f"Majority (g-mean {geomean(r_mj):.3f}×)", lw=2, ls="--")
    plt.axvline(1.0, color="#999", ls=":", lw=1)
    plt.xlabel("Latency ratio vs Oracle (×)  —  lower is better")
    plt.ylabel("CDF")
    plt.title(f"{ds.upper()} ratio CDF (K={k})")
    plt.xlim(0, max(2.0, np.percentile(np.concatenate([r_dt,r_ce,r_mj]), 99)))
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    out2 = os.path.join(models_dir, f"{ds}_latency_ratio_cdf_k{k}.png")
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"[SAVE] {out2}")

    # 3) Histogram of ratios
    fig3 = plt.figure(figsize=(7.6,5.0))
    mx = np.percentile(np.concatenate([r_dt, r_ce, r_mj]), 99.5)
    bins = np.linspace(0, max(2.0, mx), 60)
    plt.hist(r_dt, bins=bins, alpha=0.45, label="DTree", density=True)
    plt.hist(r_ce, bins=bins, alpha=0.45, label="Centers", density=True)
    plt.hist(r_mj, bins=bins, alpha=0.35, label="Majority", density=True)
    plt.axvline(1.0, color="#999", ls=":", lw=1)
    plt.xlabel("Latency ratio vs Oracle (×)")
    plt.ylabel("Density")
    plt.title(f"{ds.upper()} ratio histogram (K={k})")
    plt.legend(frameon=False)
    plt.tight_layout()
    out3 = os.path.join(models_dir, f"{ds}_latency_ratio_hist_k{k}.png")
    plt.savefig(out3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    print(f"[SAVE] {out3}")

    # 4) Scatter Oracle vs methods
    def scatter_pair(y_true_lat, y_pred_lat, name, fname):
        fig = plt.figure(figsize=(6.2,6.0))
        lim = max(np.percentile(np.concatenate([y_true_lat, y_pred_lat]), 99.5), 1.0)
        plt.scatter(y_true_lat, y_pred_lat, s=6, alpha=0.25, edgecolors="none")
        m = max(lim, 1.0)
        plt.plot([0,m],[0,m], color="#444", lw=1, ls="--")
        if args.logy:
            plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Oracle latency")
        plt.ylabel(f"{name} latency")
        gm = geomean(y_pred_lat / y_true_lat)
        plt.title(f"{ds.upper()} Oracle vs {name}  (g-mean ratio {gm:.3f}×)")
        plt.tight_layout()
        plt.savefig(fname, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVE] {fname}")
    out4a = os.path.join(models_dir, f"{ds}_scatter_oracle_vs_dtree_k{k}.png")
    out4b = os.path.join(models_dir, f"{ds}_scatter_oracle_vs_centers_k{k}.png")
    scatter_pair(lat_oracle, lat_dtree,  "DTree",  out4a)
    scatter_pair(lat_oracle, lat_center, "Centers", out4b)

    # 5) Per true-label ratio bars (geometric mean)
    groups = defaultdict(list)
    for i in range(len(y_true)):
        groups[int(y_true[i])].append(i)
    labels_sorted = sorted(groups.keys(), key=lambda l: label_name(l))
    geos_dt = []; geos_ce = []; names = []
    for l in labels_sorted:
        idx = groups[l]
        names.append(label_name(l))
        geos_dt.append(geomean(lat_dtree[idx]  / lat_oracle[idx]))
        geos_ce.append(geomean(lat_center[idx] / lat_oracle[idx]))
    x = np.arange(len(names))
    fig5 = plt.figure(figsize=(max(10, 0.6*len(names)), 5.0))
    w = 0.45
    plt.bar(x - w/2, geos_dt, width=w, label="DTree", color="#4C78A8")
    plt.bar(x + w/2, geos_ce, width=w, label="Centers", color="#F58518")
    plt.axhline(1.0, color="#999", ls=":", lw=1)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Geometric mean ratio vs Oracle (×)")
    plt.title(f"{ds.upper()} per-true-label ratios (K={k})")
    plt.legend(frameon=False)
    plt.tight_layout()
    out5 = os.path.join(models_dir, f"{ds}_label_ratio_bars_k{k}.png")
    plt.savefig(out5, dpi=180, bbox_inches="tight")
    plt.close(fig5)
    print(f"[SAVE] {out5}")

    # Console summary
    print("\n[Summary]")
    print(f"Mean cycles:  Oracle={lat_oracle.mean():.1f}, "
          f"DTree={lat_dtree.mean():.1f} ({lat_dtree.mean()/lat_oracle.mean():.3f}×), "
          f"Centers={lat_center.mean():.1f} ({lat_center.mean()/lat_oracle.mean():.3f}×), "
          f"Majority={lat_major.mean():.1f} ({lat_major.mean()/lat_oracle.mean():.3f}×)")
    print(f"Geo  cycles:  Oracle={geomean(lat_oracle):.1f}, "
          f"DTree={geomean(lat_dtree):.1f} ({geomean(r_dt):.3f}×), "
          f"Centers={geomean(lat_center):.1f} ({geomean(r_ce):.3f}×), "
          f"Majority={geomean(lat_major):.1f} ({geomean(r_mj):.3f}×)")

if __name__ == "__main__":
    main()
