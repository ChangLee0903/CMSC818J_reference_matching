python centers_decider_latency.py fit \
  --dataset fashion --split train \
  --k 32 --iters 3 \
  --tau 0.6 --lam 0.6 --medoid-cap 512 --dedup-thr 0.992

python centers_decider_latency.py predict \
  --dataset fashion --split test \
  --trained-prefix fashion_train --k 32

python viz_centers.py --trained-prefix fashion_train --k 32 \
  --dataset fashion --split train --sort count --cols 12