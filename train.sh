python train -e 20 -b 64 -o 'RMSprop' --lr 0.003 --loss 'mae' &&  # lightblue in init-logs
python train -e 20 -b 64 -o 'RMSprop' --lr 0.004 --loss 'logcosh' # darkblue in init-logs