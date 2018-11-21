import threading
import optparse

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.losses import mean_squared_error
from keras.optimizers import RMSprop

from model import mobnet

class ThreadSafeIterator:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g

@threadsafe_generator
def train_generator(df):
    while True:
        shuffle_indices = np.arange(len(df))
        shuffle_indices = np.random.permutation(shuffle_indices)
        
        for start in range(0, len(df), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(df))
            df_batch = df.iloc[shuffle_indices[start:end]]
            
            x_batch = []
            y_xbatch = df_batch['XCam']
            y_ybatch = df_batch['YCam']
            
            for _fn in df_batch['file_names']:
                img = cv2.imread('{}/{}'.format(dset_path, _fn))
                img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                
#               # === You can add data augmentations here. === #
#                 if np.random.random() < 0.5:
#                     img, mask = img[:, ::-1, :], mask[..., ::-1, :]  # random horizontal flip
                
                x_batch.append(img)
            
            yield np.asarray(x_batch), {'y_xcam': y_xbatch, 'y_ycam': y_ybatch}

if __name__ == '__main__':
    WIDTH = 224
    HEIGHT = 224
    dset_path = '../gazecapture'

    parser = optparse.OptionParser()

    parser.add_option('-e', '--epochs',
    action="store", dest="epochs",
    help="epochs", default="10")

    parser.add_option('-b', '--batch_size',
    action="store", dest="bs",
    help="b", default="16")

    options, args = parser.parse_args()

    epochs = int(options.epochs)
    BATCH_SIZE = int(options.bs)

    # testdf = pd.read_csv('test-df.csv')
    traindf = pd.read_csv('train-df.csv')
    train, val = train_test_split(traindf, test_size=0.1)

    model = mobnet((HEIGHT, WIDTH, 3))
    model.compile(loss = {'y_xcam': mean_squared_error,
                        'y_ycam': mean_squared_error},
                optimizer = RMSprop())
    #               metrics = ['accuracy', 'mae'])

    model.fit_generator(generator=train_generator(train),
                            steps_per_epoch=np.ceil(float(len(train)) / float(BATCH_SIZE)),
                            epochs=epochs,
                            verbose=1,
    #                         callbacks=callbacks,
                            validation_data=train_generator(val),
                            validation_steps=np.ceil(float(len(val)) / float(BATCH_SIZE)))