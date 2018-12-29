import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.losses import mean_squared_error, logcosh, mean_absolute_error
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

from model import mobnet
from generator import generator

dset_path = '../gazecapture'
dset_df = 'landscape-l'

shape = (375, 667, 3)
lr = 1e-3
optim = RMSprop(lr)
loss = logcosh
BATCH_SIZE = 16

test = pd.read_csv('{}-data/{}-testdf.csv'.format(dset_df, dset_df))
model = mobnet(shape, None)
model.compile(loss = loss,
            optimizer = optim,
            metrics = ['mae', 'mse'])


model.load_weights('{}-data/weights/0.002-RMSprop-logcosh.hdf5'.format(dset_df))

metrics = model.evaluate_generator(generator=generator(test, shape, BATCH_SIZE, dset_path, training=False),
                         steps=np.ceil(float(len(test)) / float(BATCH_SIZE)),
                         verbose=1)

print(model.metrics_names, metrics)
