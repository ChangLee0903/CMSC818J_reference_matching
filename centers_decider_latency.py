#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
centers_decider_latency.py  —  latency-aware k-medoids (fast, label-aware)

重點改動
- 分層初始化(label-aware)：依 best label 分佈 + 均分，保證各 (fmt,flow) 都有代表
- 溫度化指派(temperature τ)：對延遲差距放大，降低 argmin 抖動（Fashion 很有用）
- medoid 去重回補：去掉幾乎重複的中心，再用最不像現有中心的樣本補齊，避免塌縮
- 全程以 latency table 加速：T[i,l]= latency(mask_i, label_l)，assign/relabel 查表

I/O
data/cache/
  <prefix>_sparse.npz               # masks[Nh,H,W] bool, ids[Nh] object
  best_choices_<prefix>_sparse.npz  # ids, best_fmt, best_flow
models/
  <train_prefix>_centers_k{K}.pkl
  <test_prefix>_preds_from_<train_prefix>_k{K}.npz

依賴：simulation.estimate_spmv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from collections import Counter
from tqdm import tqdm
import joblib

# ---- 你的 SpMV 模擬器 ----
from simulation import estimate_spmv
from spmv_cfg import (
    FORMATS, FLOWS, SPMV_KW,       # 統一的 SpMV 參數與標籤空間
    pack_label, unpack_label, label_space
)

# 目錄
CACHE_DIR  = Path("data/cache")
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 標籤空間（不含 ELL/HYB）
FMT_LIST  = ["Dense", "CSR", "BCSR", "DIA", "COO", "LIL"]
FLOW_LIST = ["IP", "OP"]

# latency table 會用到的候選（可依需求增減）
CAND_FMTS  = ["Dense", "CSR", "BCSR", "DIA"]
CAND_FLOWS = ["IP", "OP"]

# ------------------ 工具 ------------------
def pack_label(fmt: str, flow: str) -> int:
    return FMT_LIST.index(fmt) * len(FLOW_LIST) + FLOW_LIST.index(flow)

def unpack_label(lbl: int) -> Tuple[str, str]:
    i, j = divmod(int(lbl), len(FLOW_LIST))
    return FMT_LIST[i], FLOW_LIST[j]

def label_space() -> List[int]:
    return [pack_label(f, d) for f in CAND_FMTS for d in CAND_FLOWS]

def load_sparse(prefix: str) -> Tuple[np.ndarray, List[str]]:
    p = CACHE_DIR / f"{prefix}_sparse.npz"
    if not p.exists(): raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    if "masks" not in z or "ids" not in z:
        raise ValueError(f"{p} must contain keys 'masks' and 'ids'")
    masks = z["masks"].astype(bool); ids = list(z["ids"]); z.close()
    if len(ids) != masks.shape[0]:
        raise RuntimeError(f"ids ({len(ids)}) != masks N ({masks.shape[0]})")
    return masks, ids

def load_best_labels(prefix: str) -> Dict[str, int]:
    p = CACHE_DIR / f"best_choices_{prefix}_sparse.npz"
    if not p.exists(): raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    ids, fmts, flows = list(z["ids"]), list(z["best_fmt"]), list(z["best_flow"])
    z.close()
    return {ids[i]: pack_label(fmts[i], flows[i]) for i in range(len(ids))}

# ------------------ 相似度/距離 ------------------
def dice_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Dice 相似度：2|A∩B| / (|A|+|B|)，輸出 [NX, NY]；X,Y 為 uint8 0/1 扁平化。"""
    inter = X @ Y.T
    sx = X.sum(axis=1, dtype=np.int32)[:, None]
    sy = Y.sum(axis=1, dtype=np.int32)[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((sx+sy)>0, (2.0*inter)/(sx+sy), 1.0)

# ------------------ 分層初始化 ------------------
def label_aware_seed(X_bool_flat, ids, id2y, k, labels_all, lam=0.5, seed=0):
    """
    依最佳 label 分佈 + 均分做配額：
      quota = lam * freq + (1-lam) * uniform
    每個 label 至少 1；在各自子集合內用 Dice farthest-first 取 quota 個 seed。
    """
    rng = np.random.default_rng(seed)
    N = X_bool_flat.shape[0]
    X = X_bool_flat.astype(np.uint8)
    L = len(labels_all)
    lab2idx = {int(l): i for i, l in enumerate(labels_all)}

    # 標籤索引（未知記為 -1）
    y_idx = []
    for s in ids:
        v = id2y.get(s, None)
        y_idx.append(lab2idx[int(v)] if v is not None else -1)
    y_idx = np.array(y_idx, dtype=int)

    freq = np.bincount(np.clip(y_idx, 0, L-1), minlength=L).astype(float)
    if freq.sum() == 0:
        # 沒標籤 → 退化為全域 farthest-first
        return farthest_first_init(X_bool_flat, k, metric="dice", seed=seed)

    p_freq = freq / freq.sum()
    p_uniform = np.ones(L) / L
    target = lam * p_freq + (1.0 - lam) * p_uniform
    quota = np.maximum(1, np.round(target * k).astype(int))
    while quota.sum() > k: quota[np.argmax(quota)] -= 1
    while quota.sum() < k: quota[np.argmin(quota)] += 1

    chosen = []
    for l in range(L):
        idx = np.where(y_idx == l)[0]
        if idx.size == 0: 
            continue
        q = int(min(quota[l], idx.size))
        sub = X[idx]
        first = int(rng.integers(idx.size))
        pick = [int(idx[first])]
        sims = dice_similarity(sub[[first]], sub)[0]
        dmin = 1.0 - sims
        for _ in range(1, q):
            cand_local = int(np.argmax(dmin))
            pick.append(int(idx[cand_local]))
            sims = dice_similarity(sub[[cand_local]], sub)[0]
            dmin = np.minimum(dmin, 1.0 - sims)
        chosen.extend(pick)

    chosen = np.array(sorted(set(chosen)), dtype=int)
    if chosen.size < k:
        # 用全域 farthest-first 補滿
        extra = farthest_first_init(X_bool_flat, k - chosen.size, metric="dice", seed=seed+1)
        chosen = np.concatenate([chosen, extra])
    return chosen[:k]

def farthest_first_init(X_bool_flat: np.ndarray, k: int, metric: str = "dice", seed: int = 0) -> np.ndarray:
    """傳統 farthest-first（備援用）"""
    rng = np.random.default_rng(seed)
    N, _ = X_bool_flat.shape
    X = X_bool_flat.astype(np.uint8)
    centers = [int(rng.integers(N))]

    if metric == "dice":
        sims = dice_similarity(X[[centers[0]]], X)[0]
        dmin = 1.0 - sims
        for _ in range(1, k):
            cand = int(np.argmax(dmin))
            centers.append(cand)
            sims = dice_similarity(X[[cand]], X)[0]
            dmin = np.minimum(dmin, 1.0 - sims)
    else:
        sx = X.sum(axis=1, dtype=np.int32)
        inter = (X[[centers[0]]] @ X.T)[0]
        union = sx + sx[centers[0]] - inter
        jac = np.where(union>0, inter/union, 1.0)
        dmin = 1.0 - jac
        for _ in range(1, k):
            cand = int(np.argmax(dmin))
            centers.append(cand)
            inter = (X[[cand]] @ X.T)[0]
            union = sx + sx[cand] - inter
            jac = np.where(union>0, inter/union, 1.0)
            dmin = np.minimum(dmin, 1.0 - jac)
    return np.array(centers, dtype=int)

# ------------------ Latency table ------------------
def build_latency_table(masks: np.ndarray, labels_all: List[int]) -> np.ndarray:
    """
    回傳 T: [N, L]，T[i, l] = latency(mask_i, labels_all[l]) （int32）。
    """
    N = masks.shape[0]
    L = len(labels_all)
    T = np.empty((N, L), dtype=np.int32)
    pbar = tqdm(total=N*L, desc="build latency table", leave=False)
    for i in range(N):
        A = masks[i].astype(int)
        for l, lbl in enumerate(labels_all):
            fmt, flow = unpack_label(int(lbl))
            s = estimate_spmv(A, fmt, dataflow=flow, **SPMV_KW)
            T[i, l] = int(s["latency"])
            pbar.update(1)
    pbar.close()
    return T

# ------------------ 指派（溫度化） ------------------
def assign_by_latency_table(T: np.ndarray, center_labels: np.ndarray, labels_all: List[int], tau: float = 0.6) -> np.ndarray:
    """
    先做每樣本的 min-normalize，再除以 tau（tau<1 放大差距）→ argmin。
    """
    label_to_col = {int(lbl): j for j, lbl in enumerate(labels_all)}
    cols = np.array([label_to_col[int(x)] for x in center_labels], dtype=np.int32)  # [K]
    C = T[:, cols].astype(np.float32)  # [N,K]
    m = C.min(axis=1, keepdims=True)
    Cn = (C - m) / max(1e-6, float(tau))
    return np.argmin(Cn, axis=1)

# ------------------ 中心標籤決策 ------------------
def choose_center_label_for_cluster(idx: np.ndarray,
                                    ids: List[str],
                                    id2y: Dict[str,int],
                                    T: np.ndarray,
                                    labels_all: List[int]) -> int:
    """
    先用多數決；若沒有標籤，改用 latency table 的總和最小。
    """
    ys = [id2y.get(ids[i]) for i in idx if ids[i] in id2y]
    ys = [y for y in ys if y is not None]
    if ys:
        return int(Counter(ys).most_common(1)[0][0])

    if idx.size == 0:
        return int(labels_all[0])
    total_per_label = T[idx].sum(axis=0)  # [L]
    lbest = int(np.argmin(total_per_label))
    return int(labels_all[lbest])

# ------------------ medoid：Dice 為主、可附加 latency 細微權重 ------------------
def choose_medoid_dice(idx: np.ndarray,
                       masks: np.ndarray,
                       T: np.ndarray,
                       cluster_label: int,
                       labels_all: List[int],
                       sample_cap: int = 256,
                       alpha: float = 0.0,
                       seed: int = 0) -> int:
    """
    以 Dice 距離總和為主，alpha * latency 作輕微 tie-break。回傳全域索引。
    """
    if idx.size == 0:
        return -1
    rng = np.random.default_rng(seed)
    if idx.size > sample_cap:
        cand = np.array(sorted(rng.choice(idx, size=sample_cap, replace=False)))
    else:
        cand = idx

    X = masks[idx].reshape(len(idx), -1).astype(np.uint8)     # [G,D]
    C = masks[cand].reshape(len(cand), -1).astype(np.uint8)   # [C,D]
    sims = dice_similarity(X, C)                               # [G,C]
    dice_cost = (1.0 - sims).sum(axis=0)                      # [C]

    if alpha > 0.0:
        lab2col = {int(lbl): j for j, lbl in enumerate(labels_all)}
        col = lab2col[int(cluster_label)]
        lat_cost = T[idx, col].sum(axis=0)                    # scalar
        total = dice_cost + float(alpha) * lat_cost
        m_local = int(np.argmin(total))
    else:
        m_local = int(np.argmin(dice_cost))
    return int(cand[m_local])

# ------------------ 去重 + 回補 ------------------
def dedup_and_refill(centers: np.ndarray, masks: np.ndarray, want_k: int, seed: int = 0, thr: float = 0.992) -> np.ndarray:
    """
    刪除 Dice>=thr 的幾乎重複中心，再用「最不像現有中心」的樣本回補。
    """
    rng = np.random.default_rng(seed)
    C = centers.reshape(centers.shape[0], -1).astype(np.uint8)
    inter = C @ C.T
    s = C.sum(axis=1, dtype=np.int32)
    dice = (2.0*inter) / (s[:,None] + s[None,:] + 1e-9)

    keep = []
    for i in range(C.shape[0]):
        dup = False
        for j in keep:
            if dice[i, j] >= thr:
                dup = True
                break
        if not dup:
            keep.append(i)
    keep = np.array(keep, dtype=int)
    Ckept = centers[keep]
    if Ckept.shape[0] >= want_k:
        return Ckept[:want_k]

    # 回補：選擇與現有中心相似度最高值最小的樣本（最不像）
    X = masks.reshape(masks.shape[0], -1).astype(np.uint8)
    C2 = Ckept.reshape(Ckept.shape[0], -1).astype(np.uint8)
    inter2 = X @ C2.T
    sx = X.sum(axis=1, dtype=np.int32)[:, None]
    sc = C2.sum(axis=1, dtype=np.int32)[None, :]
    diceXC = (2.0*inter2) / (sx + sc + 1e-9)
    score = diceXC.max(axis=1)  # 與現有中心的相似度
    order = np.argsort(score)   # 最不像的優先
    needed = want_k - Ckept.shape[0]
    add_idx = order[:needed]
    return np.concatenate([Ckept, masks[add_idx]], axis=0)

# ------------------ FIT ------------------
def fit_centers(prefix_train: str, k: int, iters: int,
                init_metric: str, tau: float,
                medoid_cap: int, alpha: float, lam: float,
                dedup_thr: float, seed: int) -> Path:
    masks, ids = load_sparse(prefix_train)           # [N,H,W]
    id2y = load_best_labels(prefix_train)            # {id -> packed}
    N, H, W = masks.shape
    print(f"[FIT] data={prefix_train}  N={N}  HxW={H}x{W}  K={k} iters={iters}")

    labels_all = label_space()                       # L labels
    # 1) latency table（一次性）
    T = build_latency_table(masks, labels_all)       # [N,L]

    # 2) 分層初始化（若沒標籤則退化為 farthest-first）
    Xflat = masks.reshape(N, -1).astype(bool)
    init_idx = label_aware_seed(Xflat, ids, id2y, k, labels_all, lam=lam, seed=seed)
    centers = masks[init_idx].copy()
    center_ids = [ids[i] for i in init_idx]

    # 3) 初始化中心標籤：多數決，否則取 latency-table 該樣本最小欄
    center_labels = []
    lab2col = {int(lbl): j for j, lbl in enumerate(labels_all)}
    for i_global in init_idx:
        lbl = id2y.get(ids[i_global])
        if lbl is None:
            lbest = int(np.argmin(T[i_global]))
            lbl = int(labels_all[lbest])
        center_labels.append(int(lbl))
    center_labels = np.array(center_labels, dtype=np.int32)

    # 4) 迭代（assign → relabel → medoid → 去重回補）
    for t in range(1, iters+1):
        print(f"[FIT] iter {t}/{iters} assign by latency (tau={tau}) ...")
        labels = assign_by_latency_table(T, center_labels, labels_all, tau=tau)  # [N]

        print("[FIT] relabel centers by cluster majority/latency ...")
        new_center_labels = []
        for c in range(k):
            idx = np.where(labels == c)[0]
            new_center_labels.append(
                choose_center_label_for_cluster(idx, ids, id2y, T, labels_all)
            )
        center_labels = np.array(new_center_labels, dtype=np.int32)

        print("[FIT] update medoids (Dice, capped) ...")
        new_centers, new_center_ids = [], []
        for c in tqdm(range(k), desc="medoids", leave=False):
            idx = np.where(labels == c)[0]
            m = choose_medoid_dice(idx, masks, T, center_labels[c],
                                   labels_all, sample_cap=medoid_cap,
                                   alpha=alpha, seed=seed)
            if m < 0:
                new_centers.append(centers[c]); new_center_ids.append(center_ids[c])
            else:
                new_centers.append(masks[m]);   new_center_ids.append(ids[m])
        centers = np.stack(new_centers, axis=0).astype(bool)
        center_ids = new_center_ids

        print(f"[FIT] de-duplicate & refill centers (thr={dedup_thr}) ...")
        centers = dedup_and_refill(centers, masks, want_k=k, seed=seed, thr=dedup_thr)

    out_pkl = MODELS_DIR / f"{prefix_train}_centers_k{k}.pkl"
    joblib.dump(dict(
        centers=centers.astype(bool),
        center_ids=np.array(center_ids, dtype=object),
        center_labels=center_labels.astype(np.int32),
        fmt_list=np.array(FMT_LIST, dtype=object),
        flow_list=np.array(FLOW_LIST, dtype=object),
        H=np.int32(H), W=np.int32(W),
        train_prefix=np.array(prefix_train, dtype=object),
        meta=dict(k=k, iters=iters,
                  tau=float(tau), medoid_cap=medoid_cap,
                  alpha=float(alpha), lam=float(lam), dedup_thr=float(dedup_thr),
                  seed=seed)
    ), out_pkl)
    print(f"[FIT] saved centers -> {out_pkl}")
    return out_pkl

# ------------------ PREDICT（全測試集） ------------------
def predict_with_centers(prefix_test: str, centers_pkl: Path) -> Path:
    bundle = joblib.load(centers_pkl)
    centers = bundle["centers"].astype(bool)
    center_labels = bundle["center_labels"].astype(np.int32)
    train_prefix = str(bundle["train_prefix"])
    K = centers.shape[0]

    masks_te, ids_te = load_sparse(prefix_test)
    N, H, W = masks_te.shape
    assert centers.shape[1:] == (H, W), "測試與中心尺寸不一致"

    labels_all = label_space()
    # 在整個 testing 上建 latency table（全量）
    T_te = build_latency_table(masks_te, labels_all)
    idx = assign_by_latency_table(T_te, center_labels, labels_all, tau=0.8)  # 預測時可用較溫和的 tau
    y_pred = center_labels[idx]

    # y_true 若存在（整個 testing）
    y_true = None
    bc_path = CACHE_DIR / f"best_choices_{prefix_test}_sparse.npz"
    if bc_path.exists():
        id2y = load_best_labels(prefix_test)
        y_true = np.array([id2y.get(i, -1) for i in ids_te], dtype=np.int32)

    pred_fmt, pred_flow = [], []
    for lbl in y_pred:
        f, d = unpack_label(int(lbl))
        pred_fmt.append(f); pred_flow.append(d)

    out_npz = MODELS_DIR / f"{prefix_test}_preds_from_{train_prefix}_k{K}.npz"
    np.savez_compressed(
        out_npz,
        ids=np.array(ids_te, dtype=object),
        y_pred=y_pred.astype(np.int32),
        pred_fmt=np.array(pred_fmt, dtype=object),
        pred_flow=np.array(pred_flow, dtype=object),
        y_true=(y_true if y_true is not None else np.array([], dtype=np.int32)),
        centers_path=np.array(str(centers_pkl), dtype=object)
    )
    print(f"[PREDICT] saved -> {out_npz} | N={N}")

    if y_true is not None and (y_true >= 0).any():
        acc = (y_pred == y_true).mean()
        for i in range(N):
            if y_pred[i] != y_true[i]:
                fp, dp = unpack_label(int(y_pred[i]))
                ft, dt = unpack_label(int(y_true[i]))
                sp = estimate_spmv(masks_te[i].astype(int), fp, dataflow=dp, **SPMV_KW)
                st = estimate_spmv(masks_te[i].astype(int), ft, dataflow=dt, **SPMV_KW)
                print(f"ID={ids_te[i]} | PRED=({fp},{dp})[{sp['latency']}], TRUE=({ft},{dt})[{st['latency']}]")
        print(f"[PREDICT] accuracy on FULL test: {acc*100:.2f}%  ({N} / {N})")
    return out_npz

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser("Latency-aware binary centers (k-medoids, label-aware)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_fit = sub.add_parser("fit", help="train centers on training split")
    ap_fit.add_argument("--dataset", choices=["mnist","fashion"], required=True)
    ap_fit.add_argument("--split", choices=["train"], default="train")
    ap_fit.add_argument("--k", type=int, default=32)
    ap_fit.add_argument("--iters", type=int, default=3)
    ap_fit.add_argument("--init-metric", choices=["dice","jaccard"], default="dice",
                        help="僅在無標籤或退化情況下會用到")
    ap_fit.add_argument("--tau", type=float, default=0.6, help="assignment 溫度（越小越敏感）")
    ap_fit.add_argument("--medoid-cap", type=int, default=256, help="每群 medoid 候選上限")
    ap_fit.add_argument("--alpha", type=float, default=0.0, help="medoid 選擇時 latency 的輕微權重")
    ap_fit.add_argument("--lam", type=float, default=0.5, help="分層初始化的分配權重 [0,1]")
    ap_fit.add_argument("--dedup-thr", type=float, default=0.992, help="中心去重的 Dice 門檻")
    ap_fit.add_argument("--seed", type=int, default=0)

    ap_pred = sub.add_parser("predict", help="use trained centers to predict a split (FULL test)")
    ap_pred.add_argument("--dataset", choices=["mnist","fashion"], required=True)
    ap_pred.add_argument("--split", choices=["test","train"], default="test")
    ap_pred.add_argument("--trained-prefix", required=True)
    ap_pred.add_argument("--k", type=int, required=True)

    args = ap.parse_args()
    if args.cmd == "fit":
        prefix = f"{args.dataset}_{args.split}"
        fit_centers(prefix, k=args.k, iters=args.iters,
                    init_metric=args.init_metric, tau=args.tau,
                    medoid_cap=args.medoid_cap, alpha=args.alpha,
                    lam=args.lam, dedup_thr=args.dedup_thr, seed=args.seed)
    else:
        test_prefix = f"{args.dataset}_{args.split}"
        centers_pkl = MODELS_DIR / f"{args.trained_prefix}_centers_k{args.k}.pkl"
        if not centers_pkl.exists():
            raise FileNotFoundError(centers_pkl)
        predict_with_centers(test_prefix, centers_pkl)

if __name__ == "__main__":
    main()
