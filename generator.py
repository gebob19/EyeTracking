import threading
import optparse
import cv2
import pandas as pd
import numpy as np

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
def generator(df, shape, batch_size, dset_path, shuffle=True, training=True):
    while True:
        if shuffle:
            shuffle_indices = np.arange(len(df))
            shuffle_indices = np.random.permutation(shuffle_indices)

        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            if shuffle:
                df_batch = df.iloc[shuffle_indices[start:end]]
            else:
                df_batch = df.iloc[start:end]
            
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

                if training:
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