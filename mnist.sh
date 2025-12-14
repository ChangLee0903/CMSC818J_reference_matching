python centers_decider_latency.py fit \
  --dataset mnist --split train \
  --k 32 --iters 3 \
  --tau 0.9 --lam 0.5 --medoid-cap 256 --dedup-thr 0.995

python centers_decider_latency.py predict \
  --dataset mnist --split test \
  --trained-prefix mnist_train --k 32

python viz_centers.py --trained-prefix mnist_train --k 32