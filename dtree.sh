python dtree_decider.py train --dataset mnist --split train 
python dtree_decider.py train --dataset fashion --split train 

python dtree_decider.py predict --dataset mnist --split test \
    --model models/mnist_train_dtree_table.pkl
python dtree_decider.py predict --dataset fashion --split test \
    --model models/fashion_train_dtree_table.pkl