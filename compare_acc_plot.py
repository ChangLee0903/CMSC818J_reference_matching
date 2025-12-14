import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

FMT_LIST  = ["Dense", "CSR", "BCSR", "DIA", "COO", "LIL"]
FLOW_LIST = ["IP", "OP"]

def pack_label(fmt: str, flow: str) -> int:
    return FMT_LIST.index(fmt) * len(FLOW_LIST) + FLOW_LIST.index(flow)

def load_preds(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    ids = list(z["ids"])
    y_pred = z["y_pred"].astype(np.int64)
    y_true = z["y_true"] if "y_true" in z else np.array([], dtype=np.int64)
    z.close()
    return ids, y_pred, y_true

def reconstruct_y_true_from_best(ids, best_path: str):
    if not os.path.exists(best_path):
        return None
    z = np.load(best_path, allow_pickle=True)
    ids_bc = list(z["ids"])
    fmts = list(z["best_fmt"])
    flows = list(z["best_flow"])
    z.close()
    id2lbl = {ids_bc[i]: pack_label(fmts[i], flows[i]) for i in range(len(ids_bc))}
    y = np.array([id2lbl.get(i, -1) for i in ids], dtype=np.int64)
    if (y < 0).any():
        return None
    return y

def majority_baseline(y_true: np.ndarray) -> float:
    cnt = Counter(y_true.tolist())
    most = cnt.most_common(1)[0][1]
    return most / len(y_true) if len(y_true) else 0.0

def main():
    ap = argparse.ArgumentParser("Compare DTree vs Centers accuracy and plot (MNIST/FASHION)")
    ap.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    ap.add_argument("--centers-k", type=int, default=32)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    ds = args.dataset.lower()
    models_dir = args.models_dir
    cache_dir = args.cache_dir
    k = args.centers_k

    pred_dtree = os.path.join(models_dir, f"{ds}_test_preds_from_dtree.npz")
    pred_cent  = os.path.join(models_dir, f"{ds}_test_preds_from_{ds}_train_k{k}.npz")
    best_fallback = os.path.join(cache_dir, f"best_choices_{ds}_test_sparse.npz")

    files = [pred_dtree, pred_cent]
    loaded = [load_preds(p) for p in files]

    # Prefer embedded y_true if present
    y_true = None
    ids_ref = None
    for ids, _, yt in loaded:
        if yt.size > 0 and (yt >= 0).all():
            y_true = yt
            ids_ref = ids
            break
    if y_true is None:
        ids_ref = loaded[0][0]
        y_try = reconstruct_y_true_from_best(ids_ref, best_fallback)
        if y_try is None:
            raise RuntimeError("No y_true found and cannot reconstruct from best_choices.")
        y_true = y_try

    names, accs = [], []
    for path, (ids, y_pred, _) in zip(files, loaded):
        if ids == ids_ref:
            y_pred_aligned = y_pred
        else:
            m = {iid: y for iid, y in zip(ids, y_pred)}
            y_pred_aligned = np.array([m.get(iid, -999999) for iid in ids_ref], dtype=np.int64)
            if (y_pred_aligned == -999999).any():
                missing = int((y_pred_aligned == -999999).sum())
                raise RuntimeError(f"{os.path.basename(path)} missing {missing} ids; cannot align.")
        acc = (y_pred_aligned == y_true).mean()
        names.append(os.path.basename(path))
        accs.append(acc)

    maj = majority_baseline(y_true)

    # ----- Pretty plot -----
    labels = ["DTree", "Centers", "Majority"]
    pretty = []
    for n in names:
        if "dtree" in n:
            pretty.append("DTree")
        elif "_train_k" in n:
            pretty.append("Centers")
        else:
            pretty.append(n)
    order = [pretty.index("DTree"), pretty.index("Centers")]
    values = [accs[order[0]], accs[order[1]], maj]

    plt.figure(figsize=(8, 5.2))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    xs = np.arange(len(labels))
    bars = plt.bar(xs, values, color=colors, edgecolor="#333333", linewidth=1.0)

    for x, v in zip(xs, values):
        plt.text(x, v + 0.015, f"{v*100:.2f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#222222")

    plt.xticks(xs, labels, fontsize=12)
    plt.ylim(0, 1.05)
    yticks = np.linspace(0, 1.0, 6)
    plt.yticks(yticks, [f"{t*100:.0f}%" for t in yticks], fontsize=10)
    plt.gca().yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    plt.gca().set_axisbelow(True)
    plt.title(f"{ds.upper()} test accuracy comparison", fontsize=14, pad=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.tight_layout()

    out_png = os.path.join(models_dir, f"{ds}_test_accuracy_comparison_k{k}.png")
    os.makedirs(models_dir, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    # plt.show()
    print(f"\nSaved plot: {out_png}")

    # Console summary
    print("[Summary]")
    print(f"  DTree   acc={values[0]*100:.2f}%")
    print(f"  Centers acc={values[1]*100:.2f}%")
    print(f"  Majority acc={values[2]*100:.2f}%")
    print(f"  N={len(y_true)} samples")

if __name__ == "__main__":
    main()
