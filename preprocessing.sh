python parse.py --prefix mnist_train   --thr 0.20
python parse.py --prefix mnist_test    --thr 0.20
python parse.py --prefix fashion_train --thr 0.20
python parse.py --prefix fashion_test  --thr 0.20
python generate_best_choices.py --prefix mnist_train
python generate_best_choices.py --prefix mnist_test
python generate_best_choices.py --prefix fashion_train
python generate_best_choices.py --prefix fashion_test