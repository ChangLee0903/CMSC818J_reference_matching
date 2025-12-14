# CMSC818J_reference_matchingHere’s a full `README.md` you can drop into the repo:

````markdown
# CMSC818J – Reference Matching for Sparse Format/Dataflow Selection

This repository contains the code used in my CMSC818J project on **dictionary-based reference matching** for choosing sparse matrix compression formats and dataflows (inner vs. outer product).  
The core idea is to learn a small dictionary of binary “basis” patterns and attach to each basis its empirically best (format, dataflow) pair, using a lightweight SpMV simulator.

The code lets you:

- preprocess MNIST and Fashion-MNIST into tiled binary masks  
- build an SpMV latency table for all format–dataflow pairs  
- train a **dictionary-based selector** (binary k-medoids)  
- train a **decision-tree baseline**  
- reproduce the **accuracy** and **ablation** plots used in the report

---

## 1. Environment and Dependencies

Tested with **Python 3.9+** on Linux.

Install dependencies (adjust as needed):

```bash
python -m venv .venv
source .venv/bin/activate

pip install \
  numpy \
  matplotlib \
  scikit-learn \
  tqdm \
  joblib \
  torch \
  torchvision
````

(If you already have your own environment, you mainly need `numpy`, `matplotlib`, `scikit-learn`, `tqdm`, `joblib`, and a PyTorch + torchvision install for MNIST/Fashion-MNIST.)

---

## 2. Repository Layout

Key files:

* `simulation.py` – analytic SpMV latency/utilization simulator
* `spmv_cfg.py` – format/dataflow definitions and simulator config
* `centers_decider_latency.py` – latency-aware k-medoids (dictionary) training + test
* `dtree_decider.py` – decision-tree baseline for policy selection
* `generate_best_choices.py` – offline oracle sweep to get best policy per mask
* `parse.py` – loads MNIST / Fashion-MNIST, tiles & binarizes into sparse masks
* `viz_centers.py` – visualize learned basis patterns
* `compare_acc_plot.py` – accuracy bar plots (Centers vs DTree vs Majority)
* `compare_latency_plot.py` – latency bar plots (optional)
* `plot_acc_vs_k.py` – ablation plot of accuracy vs number of centers K

Helper shell scripts:

* `preprocessing.sh` – run full preprocessing + oracle generation
* `dtree.sh` – train and evaluate decision-tree baselines
* `mnist.sh` – train and evaluate dictionary selector on MNIST (K=32)
* `fashion.sh` – train and evaluate dictionary selector on Fashion-MNIST (K=32)
* `mnist_ablation.sh` – run dictionary ablation on different K

Data and model paths (created automatically):

* `data/cache/`

  * `mnist_{train,test}_sparse.npz`
  * `fashion_{train,test}_sparse.npz`
  * `best_choices_{mnist,fashion}_{train,test}_sparse.npz`
* `models/`

  * `*_centers_k{K}.pkl` – learned dictionaries
  * decision-tree models / prediction dumps
* `models/`

  * accuracy and ablation plots (`*.png`)

---

## 3. Data Preprocessing & Oracle Generation

This step turns MNIST and Fashion-MNIST images into tiled binary masks and precomputes the per-tile oracle policy (best format+dataflow).

```bash
bash preprocessing.sh
```

What this does:

1. Calls `parse.py` for MNIST and Fashion-MNIST to:

   * download / load the raw datasets,
   * tile each image into fixed-size blocks,
   * binarize each block,
   * save `*_sparse.npz` files into `data/cache/`.

2. Runs `generate_best_choices.py`:

   * for each mask and each (format, dataflow),
     calls `estimate_spmv` in `simulation.py`,
   * picks the latency-minimizing policy as the **oracle label**,
   * saves `best_choices_*.npz` files into `data/cache/`.

You only need to run `preprocessing.sh` once (unless you change the tiling or binarization).

---

## 4. Decision-Tree Baseline

Train and evaluate a global feature-based decision tree on both datasets:

```bash
bash dtree.sh
```

This script:

* trains a decision tree on hand-engineered features (density, row/column stats, etc.),
* predicts the policy label per tile on each test set,
* writes model and prediction files under `models/` and/or `data/cache/`,
* prints test accuracy and approximate latency statistics to stdout.

---

## 5. Dictionary-Based Selector (Main Method)

### 5.1 Train and Evaluate on MNIST (K = 32)

```bash
bash mnist.sh
```

`mnist.sh` runs:

1. **Fit centers** with latency-aware k-medoids:

```bash
python centers_decider_latency.py fit \
  --dataset mnist --split train \
  --k 32 --iters 3 \
  --tau 0.9 --lam 0.5 \
  --medoid-cap 512 --dedup-thr 0.992
```

2. **Predict on the test split** using the learned dictionary:

```bash
python centers_decider_latency.py predict \
  --dataset mnist --split test \
  --trained-prefix mnist_train --k 32
```

This:

* learns 32 binary basis patterns (`models/mnist_train_centers_k32.pkl`),
* assigns a best (format, dataflow) to each basis using the simulator,
* evaluates on MNIST test tiles and prints:

  * policy-selection accuracy vs. oracle,
  * a per-ID breakdown for mis-matches with their latencies.

### 5.2 Train and Evaluate on Fashion-MNIST (K = 32)

```bash
bash fashion.sh
```

This script similarly:

* trains the dictionary on Fashion-MNIST (`k = 32`, `iters = 3`),
* evaluates on the Fashion test split,
* optionally visualizes learned centers:

```bash
python viz_centers.py --trained-prefix fashion_train --k 32 \
  --dataset fashion --split train --sort count --cols 12
```

The learned centers give interpretable basis patterns that correspond to common local shapes in the dataset.

---

## 6. Reproducing Main Accuracy Plots

Once `dtree.sh`, `mnist.sh`, and `fashion.sh` have finished, you can regenerate the accuracy comparison figures:

```bash
python compare_acc_plot.py
```

This script reads the prediction summaries for:

* **DTree** (decision tree baseline),
* **Centers** (dictionary selector with K=32),
* **Majority** (most frequent oracle policy),

and produces:

* `models/mnist_test_accuracy_comparison_k32.png`
* `models/fashion_test_accuracy_comparison_k32.png`

These are the two bar plots used in the paper.

---

## 7. Ablation: Number of Centers (K) on MNIST

To reproduce the **accuracy vs K** ablation (Figure `mnist_test_acc_vs_k.png`):

```bash
bash mnist_ablation.sh
```

This script repeatedly trains the dictionary on MNIST with different numbers of centers (e.g., K ∈ {2, 4, 8, 16, 32, 64}), evaluates on the test set, and stores the accuracies.

Then generate the plot:

```bash
python plot_acc_vs_k.py
```

This writes:

* `models/mnist_test_acc_vs_k.png`

which shows that:

* accuracy already jumps above ~80% with K=4,
* saturates around >90% for K ≥ 8,
* the DTree and Majority baselines appear as horizontal reference lines.

---

## 8. Using the Simulator Directly

You can also call the SpMV simulator manually to experiment with individual matrices or formats.

Example:

```python
from simulation import estimate_spmv

# toy 4x4 mask
import numpy as np
M = np.array([[1,0,0,0],
              [0,0,2,0],
              [0,3,0,0],
              [0,0,0,4]], dtype=float)

res = estimate_spmv(M, fmt="CSR", dataflow="IP", lanes=8)
print(res["latency"], res["utilization"])
```

The simulator returns a dict containing latency, utilization, and several breakdown fields (effective nnz, segment counts, memory vs compute cycles, etc.).
This is the same model used everywhere in the benchmarking scripts.

---

## 9. Re-running Everything from Scratch

To re-create all main results starting from an empty `data/cache/`, the shortest sequence is:

```bash
# 1) Preprocess data + build oracle labels
bash preprocessing.sh

# 2) Baseline decision tree (MNIST + Fashion)
bash dtree.sh

# 3) Dictionary selector on MNIST + Fashion (K = 32)
bash mnist.sh
bash fashion.sh

# 4) Accuracy plots (MNIST + Fashion)
python compare_acc_plot.py

# 5) MNIST ablation over K
bash mnist_ablation.sh
python plot_acc_vs_k.py

```

After these steps you should have:

* accuracy bar plots for both datasets,
* the MNIST accuracy-vs-K ablation figure