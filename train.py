import threading
import optparse

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.losses import mean_squared_error, logcosh, mean_absolute_error
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

from model import mobnet
from generator import generator

if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option('-e', '--epochs',
    action="store", dest="epochs",
    help="epochs", default="10")

    parser.add_option('-b', '--batch_size',
    action="store", dest="bs",
    help="b", default="16")

    parser.add_option('-s', '--shape',
    action="store", dest="s",
    help="s", default="(224, 224, 3)")

    parser.add_option('-l', '--lr',
    action="store", dest="lr",
    help="lr", default="3e-3")

    parser.add_option('-o', '--optimizer',
    action="store", dest="o",
    help="o", default="rmsprop")

    parser.add_option('-k', '--loss',
    action="store", dest="loss",
    help="s", default="mse")

    options, args = parser.parse_args()

    dset_path = '../gazecapture'
    epochs = int(options.epochs)
    BATCH_SIZE = int(options.bs)
    shape = eval(options.s)
    lr = float(options.lr)
    optimizer = options.o
    model_loss = options.loss
    
    optim = RMSprop(lr)
    if optimizer == 'Adam': 
        optim = Adam(lr) 
    loss = logcosh

    fn = 'landscape-l'
    test_df = pd.read_csv('{}-data/{}-testdf.csv'.format(fn, fn))
    train = pd.read_csv('{}-data/{}-traindf.csv'.format(fn, fn))
    test, val = train_test_split(test_df, test_size=0.1)

    model = mobnet(shape, None)

    model.compile(loss = loss,
                optimizer = optim,
                metrics = ['mae'])

    # model.load_weights('{}-data/weights/0.002-RMSprop-logcosh.hdf5'.format(fn))

    model_name = '{}-{}-{}-{}'.format(fn, lr, optimizer, model_loss)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        verbose=1,
                        min_delta=1e-5),
        TensorBoard(log_dir='./{}-data/logs/{}'.format(fn, model_name),
                    batch_size=BATCH_SIZE),
    ]
    
    callbacks.append(ModelCheckpoint(monitor='val_loss',
                    filepath='./{}-data/weights/{}-{}-{}.hdf5'.format(fn, lr, optimizer, model_loss),
                    save_best_only=True,
                    verbose=1))

    model.fit_generator(generator=generator(train, shape, BATCH_SIZE, dset_path),
                            steps_per_epoch=np.ceil(float(len(train)) / float(BATCH_SIZE)),
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=generator(val, shape, BATCH_SIZE, dset_path, training=False),
                            validation_steps=np.ceil(float(len(val)) / float(BATCH_SIZE)))
