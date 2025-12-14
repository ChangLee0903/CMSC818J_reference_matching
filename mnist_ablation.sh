python centers_decider_latency.py fit \
  --dataset mnist --split train \
  --k 2 --iters 3 \
  --tau 0.9 --lam 0.5 --medoid-cap 256 --dedup-thr 0.995


python centers_decider_latency.py fit \
  --dataset mnist --split train \
  --k 4 --iters 3 \
  --tau 0.9 --lam 0.5 --medoid-cap 256 --dedup-thr 0.995


python centers_decider_latency.py fit \
  --dataset mnist --split train \
  --k 8 --iters 3 \
  --tau 0.9 --lam 0.5 --medoid-cap 256 --dedup-thr 0.995


python centers_decider_latency.py fit \
  --dataset mnist --split train \
  --k 16 --iters 3 \
  --tau 0.9 --lam 0.5 --medoid-cap 256 --dedup-thr 0.995


python centers_decider_latency.py fit \
  --dataset mnist --split train \
  --k 64 --iters 3 \
  --tau 0.9 --lam 0.5 --medoid-cap 256 --dedup-thr 0.995

python centers_decider_latency.py predict \
  --dataset mnist --split test \
  --trained-prefix mnist_train --k 2

python centers_decider_latency.py predict \
  --dataset mnist --split test \
  --trained-prefix mnist_train --k 4

python centers_decider_latency.py predict \
  --dataset mnist --split test \
  --trained-prefix mnist_train --k 8

python centers_decider_latency.py predict \
  --dataset mnist --split test \
  --trained-prefix mnist_train --k 16

python centers_decider_latency.py predict \
  --dataset mnist --split test \
  --trained-prefix mnist_train --k 64