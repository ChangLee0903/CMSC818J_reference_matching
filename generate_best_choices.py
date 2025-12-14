#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_best_choices.py
- 僅使用 data/cache/<prefix>_sparse.npz 的 masks 來評估最佳 (fmt, dataflow)
- 產出 data/cache/best_choices_<prefix>_sparse.npz
- IDs 與 *_sparse.npz 完全相同（長度/順序）
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
from simulation import estimate_spmv  # 你已經有的 estimator

CACHE_DIR = Path("data/cache")

LANES = 8
DENSE_THRESHOLD = 0.6
BCSR_BLOCK = 4
FORMATS = ["Dense", "CSR", "BCSR", "DIA", "COO", "LIL"]   # 不含 HYB/ELL
FLOWS = ["IP", "OP"]

def pick_best_for_mask(mask_bool: np.ndarray):
    """回傳 (fmt, flow, latency, util) 中 latency 最小者"""
    A = mask_bool.astype(int)
    best = None; best_lat = None
    for fmt in FORMATS:
        for df in FLOWS:
            s = estimate_spmv(
                A, fmt,
                lanes=LANES, dataflow=df,
                block_size=BCSR_BLOCK, dense_threshold=DENSE_THRESHOLD
            )
            lat = int(s["latency"])
            util = float(s.get("util_useful", s.get("utilization", 0.0)))
            if best_lat is None or lat < best_lat:
                best_lat = lat
                best = (fmt, df, lat, util)
    return best

def process_prefix(prefix: str):
    sp_path = CACHE_DIR / f"{prefix}_sparse.npz"
    if not sp_path.exists():
        raise FileNotFoundError(sp_path)

    z = np.load(sp_path, allow_pickle=True)  # ids 是 object dtype
    if "masks" not in z or "ids" not in z:
        raise ValueError(f"{sp_path.name} must contain 'masks' and 'ids'")
    masks = z["masks"]; ids = list(z["ids"])
    z.close()

    N, H, W = masks.shape
    fmts, flows, lats, utils = [], [], [], []
    print(f"[best] {sp_path.name}: N={N}  HxW={H}x{W}")

    for i in range(N):
        fmt, flow, lat, util = pick_best_for_mask(masks[i])
        fmts.append(fmt); flows.append(flow); lats.append(lat); utils.append(util)
        if (i+1) % 1000 == 0:
            print(f"  .. {i+1}/{N}")

    out_path = CACHE_DIR / f"best_choices_{prefix}_sparse.npz"
    np.savez_compressed(
        out_path,
        ids=np.array(ids, dtype=object),          # 與 sparse 完全一致
        shapes=np.tile(np.array([[H, W]], dtype=np.int32), (N,1)),
        best_fmt=np.array(fmts, dtype=object),
        best_flow=np.array(flows, dtype=object),
        best_latency=np.array(lats, dtype=np.int64),
        best_util=np.array(utils, dtype=np.float32),
        lanes=np.int32(LANES),
        block_size=np.int32(BCSR_BLOCK),
        dense_threshold=np.float32(DENSE_THRESHOLD),
        formats=np.array(FORMATS, dtype=object),
        flows=np.array(FLOWS, dtype=object),
        source=np.array(str(sp_path), dtype=object),
    )
    print(f"  -> wrote {out_path.name}")

def main():
    ap = argparse.ArgumentParser("Generate best choices from *_sparse.npz")
    ap.add_argument("--prefix", type=str, required=True,
                    help="file stem in data/cache, e.g. mnist_train / fashion_test")
    args = ap.parse_args()
    process_prefix(args.prefix)

if __name__ == "__main__":
    main()
