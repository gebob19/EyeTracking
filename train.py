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
def train_generator(df, shape):
    while True:
        shuffle_indices = np.arange(len(df))
        shuffle_indices = np.random.permutation(shuffle_indices)
        
        for start in range(0, len(df), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(df))
            df_batch = df.iloc[shuffle_indices[start:end]]
            
            x_batch = []
            y_batch = []
            
            xcam = df_batch['XCam'].values
            ycam = df_batch['YCam'].values
            faceh = df_batch['FaceH'].values
            faceW = df_batch['FaceW'].values
            faceX = df_batch['FaceX'].values
            faceY = df_batch['FaceY'].values
            isValid = df_batch['IsValid'].values
            
            for index, _fn in enumerate(df_batch['file_names']):
                img = cv2.imread('{}/{}'.format(dset_path, _fn))
                img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

                # data augmentation
                if np.random.random() < 0.5 and isValid[index]:
                    fh, fw, fx, fy = faceh[index], faceW[index], faceX[index], faceY[index]
                    # sample amount of noise from distribution
                    noise_amount = np.random.uniform(low=5, high=100)
                    noise = np.random.randint(noise_amount, size=shape, dtype='uint8')
                    # zero noise from face
                    noise[int(fy):int(fy+fh), int(fx):int(fx+fw), :] = 0
                    img = img + noise

                y_batch.append([xcam[index], ycam[index]])
                x_batch.append(img)
            
            yield np.asarray(x_batch), np.asarray(y_batch)

if __name__ == '__main__':
    dset_path = '../gazecapture'

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

    # testdf = pd.read_csv('test-df.csv')
    train = pd.read_csv('landscape-traindf.csv')
    val = pd.read_csv('landscape-valdf.csv')

    # train, val = train_test_split(traindf, test_size=0.1)

    model_name = "basemodel_{}-lr:{}-bs:{}-loss:{}-{}".format(shape, lr, BATCH_SIZE, model_loss, optimizer)
    model = mobnet(shape, None)

    model.compile(loss = loss,
                optimizer = optim,
                metrics = ['mae'])

    # model.load_weights('logcosh-0.001-RMSprop-weights.hdf5')

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        verbose=1,
                        min_delta=1e-5),
        TensorBoard(log_dir='./stage1-logs/{}'.format(model_name),
                    batch_size=BATCH_SIZE)
    ]

    for i in range(3):
        train_sample = train.sample(frac=0.3)
        val_sample = val.sample(frac=0.3)

        callbacks.append(ModelCheckpoint(monitor='val_loss',
                        filepath='{}-{}-{}-weights-{}.hdf5'.format(model_loss, lr, optimizer, i),
                        save_best_only=True,
                        verbose=1))

        model.fit_generator(generator=train_generator(train_sample, shape),
                                steps_per_epoch=np.ceil(float(len(train_sample)) / float(BATCH_SIZE)),
                                epochs=epochs,
                                verbose=1,
                                callbacks=callbacks,
                                validation_data=train_generator(val_sample, shape),
                                validation_steps=np.ceil(float(len(val_sample)) / float(BATCH_SIZE)))
        
        callbacks = callbacks[:-1]
