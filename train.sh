python train.py -e 5 -b 64 -o 'Adam' --lr 0.003 --loss 'logcosh' &&
python train.py -e 5 -b 64 -o 'RMSprop' --lr 0.003 --loss 'logcosh' &&
python train.py -e 5 -b 64 -o 'Adam' --lr 0.001 --loss 'mae' &&
python train.py -e 5 -b 64 -o 'RMSprop' --lr 0.001 --loss 'mae' &&
python train.py -e 5 -b 64 -o 'Adam' --lr 0.003 --loss 'mse' &&
python train.py -e 5 -b 64 -o 'RMSprop' --lr 0.003 --loss 'mse'
