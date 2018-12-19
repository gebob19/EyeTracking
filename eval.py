import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.losses import mean_squared_error, logcosh, mean_absolute_error
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

from model import mobnet
from generator import train_generator

dset_path = '../gazecapture'
dset_df = 'portrait'

shape = (375, 667, 3)
lr = 1e-3
optim = RMSprop(lr)
loss = logcosh
BATCH_SIZE = 16

test = pd.read_csv('{}-test-df.csv'.format(dset_df))
model = mobnet(shape, None)
model.compile(loss = loss,
            optimizer = optim,
            metrics = ['mae', 'mse'])


# model.load_weights('logcosh-0.001-RMSprop-weights.hdf5')

metrics = model.evaluate_generator(generator=train_generator(test, shape, BATCH_SIZE, dset_path),
                         steps=np.ceil(float(len(test)) / float(BATCH_SIZE)),
                         verbose=1)

print(model.metrics_names, metrics)