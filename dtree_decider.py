#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtree_decider.py
- 僅使用 *_sparse.npz 與 best_choices_*_sparse.npz
- 嚴格對齊 IDs（完全一致）
- train: 在 training split 訓練並存模型
- predict: 用已存模型對 test split 做預測（不重訓）
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

CACHE_DIR = Path("data/cache")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# label space（與 best-choices 一致；不含 ELL/HYB）
FMT_LIST = ["Dense", "CSR", "BCSR", "DIA", "COO", "LIL"]
FLOW_LIST = ["IP", "OP"]

def pack_label(fmt: str, flow: str) -> int:
    return FMT_LIST.index(fmt) * len(FLOW_LIST) + FLOW_LIST.index(flow)

def unpack_label(lbl: int) -> tuple[str, str]:
    i, j = divmod(lbl, len(FLOW_LIST))
    return FMT_LIST[i], FLOW_LIST[j]

# ---------- features ----------
def _safe_mean(a: np.ndarray) -> float: return float(a.mean()) if a.size else 0.0
def _safe_var(a: np.ndarray) -> float:  return float(a.var()) if a.size else 0.0

def _count_true_diagonals(mask: np.ndarray) -> int:
    r, c = np.nonzero(mask)
    return int(np.unique(c - r).size) if r.size else 0

def _num_diagonals_total(M: int, N: int) -> int: return M + N - 1 if M and N else 0

def _connected_components_4(mask: np.ndarray) -> int:
    H, W = mask.shape
    if H == 0 or W == 0: return 0
    vis = np.zeros_like(mask, bool)
    cnt = 0
    for i in range(H):
        if not mask[i].any(): continue
        for j in range(W):
            if mask[i, j] and not vis[i, j]:
                cnt += 1
                stack = [(i, j)]
                vis[i, j] = True
                while stack:
                    x, y = stack.pop()
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not vis[nx, ny]:
                            vis[nx, ny] = True
                            stack.append((nx, ny))
    return cnt

def _avg_nz_neighbors(mask: np.ndarray, use_8: bool = True) -> float:
    H, W = mask.shape
    r, c = np.nonzero(mask)
    K = r.size
    if K == 0: return 0.0
    shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] if use_8 \
             else [(-1,0),(1,0),(0,-1),(0,1)]
    total = 0
    for i, j in zip(r, c):
        n = 0
        for dx, dy in shifts:
            x, y = i+dx, j+dy
            if 0 <= x < H and 0 <= y < W and mask[x, y]: n += 1
        total += n
    return total / K

FEATURE_NAMES = [
    "M","N","NNZ","Ndiags","NTdiags_ratio",
    "aver_RD","max_RD","min_RD","dev_RD",
    "aver_CD","max_CD","min_CD","dev_CD",
    "ER_DIA","ER_RD","ER_CD",
    "row_bounce","col_bounce",
    "d","cv","max_mu",
    "blocks","mean_neighbor"
]

def extract_table_features(mask: np.ndarray) -> np.ndarray:
    M, N = mask.shape
    total = M * N if M and N else 1
    NNZ = int(mask.sum())

    row_len = mask.sum(axis=1).astype(np.int32) if M else np.zeros((0,), np.int32)
    col_len = mask.sum(axis=0).astype(np.int32) if N else np.zeros((0,), np.int32)
    aver_RD = _safe_mean(row_len); max_RD = int(row_len.max()) if row_len.size else 0
    min_RD = int(row_len.min()) if row_len.size else 0; dev_RD = (_safe_var(row_len)) ** 0.5
    aver_CD = _safe_mean(col_len); max_CD = int(col_len.max()) if col_len.size else 0
    min_CD = int(col_len.min()) if col_len.size else 0; dev_CD = (_safe_var(col_len)) ** 0.5

    Ndiags = _count_true_diagonals(mask)
    NTdiags_total = _num_diagonals_total(M, N)
    NTdiags_ratio = (Ndiags / NTdiags_total) if NTdiags_total > 0 else 0.0

    ER_DIA = (NNZ / (M * max(1, Ndiags))) if M > 0 else 0.0
    ER_RD  = (NNZ / (M * max(1, max_RD))) if M > 0 else 0.0
    ER_CD  = (NNZ / (N * max(1, max_CD))) if N > 0 else 0.0

    row_bounce = float(np.mean(np.abs(np.diff(row_len)))) if row_len.size >= 2 else 0.0
    col_bounce = float(np.mean(np.abs(np.diff(col_len)))) if col_len.size >= 2 else 0.0

    d = NNZ / total
    cv = (dev_RD / aver_RD) if aver_RD > 0 else 0.0
    max_mu = float(max_RD) - float(aver_RD)
    blocks = float(_connected_components_4(mask))
    mean_neighbor = _avg_nz_neighbors(mask, use_8=True)

    return np.array([
        float(M), float(N), float(NNZ), float(Ndiags), float(NTdiags_ratio),
        float(aver_RD), float(max_RD), float(min_RD), float(dev_RD),
        float(aver_CD), float(max_CD), float(min_CD), float(dev_CD),
        float(ER_DIA), float(ER_RD), float(ER_CD),
        float(row_bounce), float(col_bounce),
        float(d), float(cv), float(max_mu),
        float(blocks), float(mean_neighbor)
    ], dtype=np.float32)

# ---------- dataset builder ----------
def build_dataset(prefix: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    prefix: e.g. 'mnist_train' / 'mnist_test' / 'fashion_train' ...
    嚴格對齊：
      - 讀 data/cache/<prefix>_sparse.npz （masks, ids）
      - 讀 data/cache/best_choices_<prefix>_sparse.npz （best_fmt/best_flow/ids）
      - 兩邊 ids 必須完全一致（長度 & 順序）
    """
    sp_path = CACHE_DIR / f"{prefix}_sparse.npz"
    bc_path = CACHE_DIR / f"best_choices_{prefix}_sparse.npz"

    if not sp_path.exists(): raise FileNotFoundError(sp_path)
    if not bc_path.exists(): raise FileNotFoundError(bc_path)

    z = np.load(sp_path, allow_pickle=True)
    masks = z["masks"]; ids_sp = list(z["ids"])
    z.close()

    bc = np.load(bc_path, allow_pickle=True)
    ids_bc   = list(bc["ids"])
    best_fmt = list(bc["best_fmt"])
    best_flow= list(bc["best_flow"])
    bc.close()

    if len(ids_sp) != len(ids_bc) or ids_sp != ids_bc:
        raise RuntimeError(
            f"ID mismatch between {sp_path.name} and {bc_path.name}.\n"
            f"  sparse.N={len(ids_sp)}  best.N={len(ids_bc)}\n"
            f"  (確保 best_choices 是用 generate_best_choices.py 對應同一份 *_sparse.npz 產生)"
        )

    X = np.vstack([extract_table_features(masks[i]) for i in range(masks.shape[0])]).astype(np.float32)
    y = np.array([pack_label(best_fmt[i], best_flow[i]) for i in range(len(ids_sp))], dtype=np.int64)
    return X, y, ids_sp

# ---------- train / predict ----------
def cmd_train(dataset: str, split: str):
    prefix = f"{dataset}_{split}"
    X, y, _ = build_dataset(prefix)

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=0
    )
    clf.fit(X, y)

    model_path = MODELS_DIR / f"{prefix}_dtree_table.pkl"
    joblib.dump(dict(
        model=clf,
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        fmt_list=np.array(FMT_LIST, dtype=object),
        flow_list=np.array(FLOW_LIST, dtype=object)
    ), model_path)
    print(f"[TRAIN] saved -> {model_path} | N={len(y)}  dims={X.shape[1]}")

def cmd_predict(dataset: str, split: str, model_path: Path):
    if not model_path.exists(): raise FileNotFoundError(model_path)
    bundle = joblib.load(model_path)
    clf: DecisionTreeClassifier = bundle["model"]

    prefix = f"{dataset}_{split}"
    X, y_true, ids = build_dataset(prefix)

    y_pred = clf.predict(X)
    pred_fmt = []; pred_flow=[]
    for lbl in y_pred:
        f, d = unpack_label(int(lbl)); pred_fmt.append(f); pred_flow.append(d)

    out_npz = MODELS_DIR / f"{prefix}_preds_from_dtree.npz"
    np.savez_compressed(
        out_npz,
        ids=np.array(ids, dtype=object),
        y_true=y_true.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
        pred_fmt=np.array(pred_fmt, dtype=object),
        pred_flow=np.array(pred_flow, dtype=object),
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        model_path=np.array(str(model_path), dtype=object),
    )
    acc = (y_true == y_pred).mean()
    print(f"[PREDICT] saved -> {out_npz} | acc={acc*100:.2f}%")

def main():
    ap = argparse.ArgumentParser("Decision tree decider (sparse only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train", help="train on training split and save model")
    ap_tr.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    ap_tr.add_argument("--split", choices=["train"], default="train")

    ap_pr = sub.add_parser("predict", help="predict on split using saved model")
    ap_pr.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    ap_pr.add_argument("--split", choices=["test","train"], default="test")
    ap_pr.add_argument("--model", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args.dataset, args.split)
    else:
        cmd_predict(args.dataset, args.split, args.model)

if __name__ == "__main__":
    main()
