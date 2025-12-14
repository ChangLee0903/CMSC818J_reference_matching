#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_latency.py  (enhanced)
Compute SpMV latency for prediction results (NOT accuracy) + sanity checks.

New in this version:
- If --compare-best:
  * Compute accuracy (pred vs oracle)
  * Show confusion (top pairs)
  * Optional --recompute-best: ignore best_choices_*.npz and recompute oracle-best by simulator
  * Emit warnings if predicted choices look identical to oracle (>99% match)

- More integrity checks and clearer prints.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import csv
from collections import defaultdict, Counter

# your simulator
from simulation import estimate_spmv
from spmv_cfg import (
    FORMATS, FLOWS, SPMV_KW,       # 統一的 SpMV 參數與標籤空間
    pack_label, unpack_label, label_space
)

CACHE_DIR  = Path("data/cache")
MODELS_DIR = Path("models")

FMT_LIST  = ["Dense", "CSR", "BCSR", "DIA", "COO", "LIL"]
FLOW_LIST = ["IP", "OP"]

def load_sparse(prefix: str):
    p = CACHE_DIR / f"{prefix}_sparse.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    if "masks" not in z or "ids" not in z:
        raise ValueError(f"{p} must contain 'masks' and 'ids'")
    masks = z["masks"].astype(bool)
    ids    = list(z["ids"])
    z.close()
    if len(ids) != masks.shape[0]:
        raise RuntimeError(f"{p}: ids({len(ids)}) != masks N({masks.shape[0]})")
    return masks, ids

def load_preds(pred_path: Path):
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    z = np.load(pred_path, allow_pickle=True)
    req = ["ids","pred_fmt","pred_flow"]
    for k in req:
        if k not in z:
            raise ValueError(f"{pred_path} missing key '{k}'")
    ids = list(z["ids"])
    pred_fmt  = [str(x) for x in list(z["pred_fmt"])]
    pred_flow = [str(x) for x in list(z["pred_flow"])]
    # optional
    y_true = z["y_true"] if "y_true" in z else None
    z.close()
    return ids, pred_fmt, pred_flow, y_true

def load_best_labels(prefix: str):
    p = CACHE_DIR / f"best_choices_{prefix}_sparse.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    req = ["ids","best_fmt","best_flow"]
    for k in req:
        if k not in z:
            raise ValueError(f"{p} missing key '{k}'")
    ids  = list(z["ids"])
    fmts = [str(x) for x in list(z["best_fmt"])]
    flows= [str(x) for x in list(z["best_flow"])]
    z.close()
    return dict((ids[i], (fmts[i], flows[i])) for i in range(len(ids)))

def percentile(a: np.ndarray, q: float) -> float:
    if a.size == 0: return 0.0
    return float(np.quantile(a, q/100.0))

def summarize(lat: np.ndarray, title: str):
    lat = lat.astype(np.float64)
    print(f"\n[{title}] N={lat.size:,}")
    print(f"  mean:   {lat.mean():,.2f}")
    print(f"  median: {percentile(lat,50):,.2f}")
    print(f"  p90:    {percentile(lat,90):,.2f}")
    print(f"  p99:    {percentile(lat,99):,.2f}")
    print(f"  sum:    {lat.sum():,.2f}")

def recompute_oracle_best_for_one(mask_bool: np.ndarray) -> tuple[str,str,int]:
    """Scan all (fmt,flow) from FMT_LIST × FLOW_LIST and pick min latency."""
    best_lat = None
    best = ("", "")
    for f in FMT_LIST:
        for d in FLOW_LIST:
            s = estimate_spmv(mask_bool.astype(int), f, dataflow=d)
            lat = int(s["latency"])
            if (best_lat is None) or (lat < best_lat):
                best_lat = lat; best = (f, d)
    return best[0], best[1], int(best_lat if best_lat is not None else -1)

def main():
    ap = argparse.ArgumentParser("Measure SpMV latency for predicted (fmt,flow) with sanity checks")
    ap.add_argument("preds_npz", type=Path, help="models/<prefix>_preds*.npz")
    ap.add_argument("--dataset", choices=["mnist","fashion"], required=True)
    ap.add_argument("--split",   choices=["test","train"],  required=True)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--dense-thr", type=float, default=0.6)

    ap.add_argument("--compare-best", action="store_true",
                    help="also compute oracle-best latency (from best_choices or recomputed)")
    ap.add_argument("--recompute-best", action="store_true",
                    help="ignore best_choices_*.npz and recompute oracle-best by simulator")

    ap.add_argument("--save-npz", action="store_true",
                    help="save per-sample latencies npz next to preds")
    ap.add_argument("--csv", action="store_true",
                    help="dump a CSV with ids, pred_fmt, pred_flow, latency[, best_latency, gap]")
    ap.add_argument("--print-each", action="store_true",
                    help="print per-sample pred/oracle fmt/flow and latency")
    ap.add_argument("--head", type=int, default=0,
                    help="only print first N rows with --print-each (0 = all)")

    args = ap.parse_args()
    prefix = f"{args.dataset}_{args.split}"

    # 1) load preds & masks
    pred_ids, pred_fmt, pred_flow, _ = load_preds(args.preds_npz)
    masks, mask_ids = load_sparse(prefix)
    id2idx = {s:i for i,s in enumerate(mask_ids)}

    # align order by preds ids; fail fast on missing
    idxs = []
    miss = []
    for s in pred_ids:
        i = id2idx.get(s, -1)
        if i < 0: miss.append(s)
        else: idxs.append(i)
    if miss:
        raise RuntimeError(f"{len(miss)} ids from preds not found in {prefix}_sparse.npz (e.g., {miss[:3]})")
    idxs = np.array(idxs, dtype=int)

    # 2) compute latency for each sample with predicted (fmt, flow)
    lat_pred = np.empty(len(pred_ids), dtype=np.int64)
    cls_buckets = defaultdict(list)
    for k, (i, f, d) in enumerate(zip(idxs, pred_fmt, pred_flow)):
        s = estimate_spmv(
            masks[i].astype(int), str(f),
            dataflow=str(d),
            lanes=int(args.lanes),
            block_size=int(args.block_size),
            dense_threshold=float(args.dense_thr)
        )
        lat = int(s["latency"])
        lat_pred[k] = lat
        cls_buckets[f"{f}_{d}"].append(lat)

    # 3) oracle best (either load or recompute)
    lat_best = None
    best_fmts = None
    best_flows = None
    if args.compare_best or args.print_each:
        best_fmts = [None]*len(pred_ids)
        best_flows= [None]*len(pred_ids)
        lat_best  = np.empty(len(pred_ids), dtype=np.int64)

        if args.recompute_best:
            # recompute from simulator → slow but ground-truth oracle
            for k, i in enumerate(idxs):
                bf, bd, bl = recompute_oracle_best_for_one(masks[i])
                best_fmts[k], best_flows[k], lat_best[k] = bf, bd, bl
        else:
            # read from file
            id2best = load_best_labels(prefix)
            for k, sid in enumerate(pred_ids):
                bf, bd = id2best.get(sid, (None, None))
                best_fmts[k] = bf
                best_flows[k] = bd
                if bf is None:
                    lat_best[k] = -1
                else:
                    s = estimate_spmv(
                        masks[idxs[k]].astype(int), str(bf),
                        dataflow=str(bd),
                        lanes=int(args.lanes),
                        block_size=int(args.block_size),
                        dense_threshold=float(args.dense_thr)
                    )
                    lat_best[k] = int(s["latency"])

    # 3.5) optional per-sample print
    if args.print_each:
        n = len(pred_ids) if args.head <= 0 else min(args.head, len(pred_ids))
        print("\n[id] pred_fmt pred_flow pred_lat   |   oracle_fmt oracle_flow oracle_lat   gap")
        for t in range(n):
            sid = pred_ids[t]
            pf, pd = pred_fmt[t], pred_flow[t]
            pl = int(lat_pred[t])
            bf = best_fmts[t] if best_fmts is not None else None
            bd = best_flows[t] if best_flows is not None else None
            bl = int(lat_best[t]) if lat_best is not None else -1
            gap = (pl - bl) if (lat_best is not None and bl >= 0) else ""
            print(f"{sid}  {pf:>7} {pd:>3} {pl:>8}   |   {str(bf):>7} {str(bd):>3} {str(bl):>10}   {gap}")

    # 4) summary
    print(f"\nMeasured latency using predicted (fmt,flow) for {len(lat_pred):,} samples.")
    summarize(lat_pred, "Predicted latency")

    print("\n[Per predicted class] mean latency (and count)")
    for cls, vals in sorted(cls_buckets.items(), key=lambda x: -len(x[1])):
        arr = np.array(vals, dtype=np.float64)
        print(f"  {cls:>8} : mean {arr.mean():,.2f}  N={len(arr):,}")

    # 5) compare to oracle: accuracy + confusion + latency gap
    if lat_best is not None:
        valid = lat_best >= 0
        summarize(lat_best[valid], "Oracle-best latency")

        # accuracy
        # 判定：pred_fmt/flow 是否與 oracle_fmt/flow 一致
        eq = np.array([
            (pred_fmt[i] == best_fmts[i]) and (pred_flow[i] == best_flows[i])
            for i in range(len(pred_ids))
        ], dtype=bool)
        acc = float(eq[valid].mean()) if valid.any() else 0.0
        print(f"\n[Label accuracy vs. oracle] on {valid.sum():,} matched samples: {acc*100:.2f}%")

        # 混淆（只列 top-10）
        pairs = Counter()
        for i in range(len(pred_ids)):
            if best_fmts[i] is None: continue
            pairs[(f"{pred_fmt[i]}_{pred_flow[i]}", f"{best_fmts[i]}_{best_flows[i]}")] += 1
        print("\n[Top-10 confusions] pred -> oracle (count)")
        for (p, o), v in pairs.most_common(10):
            print(f"  {p:>12} -> {o:<12} : {v}")

        # latency gap
        gap = (lat_pred[valid] - lat_best[valid]).astype(np.float64)
        print(f"\n[Gap vs. best] over {valid.sum():,} matched samples")
        print(f"  mean gap: {gap.mean():,.2f}")
        print(f"  median:   {percentile(gap,50):,.2f}")
        print(f"  p90:      {percentile(gap,90):,.2f}")
        print(f"  % within 0: {(gap<=0).mean()*100:5.2f}%")
        print(f"  % within 5%: {( (lat_pred[valid] <= 1.05*lat_best[valid]).mean()*100 ):5.2f}%")

        # 強烈相同 → 直接給 Warning
        if valid.any() and (acc >= 0.99):
            print("\n[WARNING] Predicted labels are ≥99% identical to oracle best.")
            print("  • Double-check your prediction pipeline: Are you training/evaluating on oracle labels?")
            print("  • Or are you accidentally writing oracle choices into pred_fmt/pred_flow?")
            print("  • Try adding --recompute-best to make sure oracle is independently recomputed.")

    # 6) optional save
    if args.save_npz:
        out_npz = args.preds_npz.with_name(args.preds_npz.stem.replace("_preds", "_latency_from_preds") + ".npz")
        np.savez_compressed(
            out_npz,
            ids=np.array(pred_ids, dtype=object),
            pred_fmt=np.array(pred_fmt, dtype=object),
            pred_flow=np.array(pred_flow, dtype=object),
            latency_pred=lat_pred.astype(np.int64),
            latency_best=(lat_best.astype(np.int64) if lat_best is not None else np.array([], dtype=np.int64))
        )
        print(f"\n[WRITE] saved per-sample latencies -> {out_npz}")

    if args.csv:
        out_csv = args.preds_npz.with_suffix(".latency.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            hdr = ["id","pred_fmt","pred_flow","latency"]
            if lat_best is not None: hdr += ["oracle_fmt","oracle_flow","oracle_latency","gap"]
            w.writerow(hdr)
            for i,sid in enumerate(pred_ids):
                row = [sid, pred_fmt[i], pred_flow[i], int(lat_pred[i])]
                if lat_best is not None:
                    bf, bd = best_fmts[i], best_flows[i]
                    bl = int(lat_best[i])
                    row += [bf, bd, bl, (int(lat_pred[i])-bl) if bl>=0 else ""]
                w.writerow(row)
        print(f"[WRITE] CSV -> {out_csv}")

if __name__ == "__main__":
    main()
