import tensorflow as tf
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model
from keras.losses import mean_squared_error
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten

# try pretrained weights and default config to get a baseline
shape = (224, 224, 3)
alpha = 1
depth_multiplier = 1

def mobilnet():
    model_in = MobileNetV2(input_shape=shape,
    #                     alpha=alpha,
    #                     depth_multiplier=depth_multiplier,
                        include_top=False,
                        weights='imagenet',
                        pooling=None)

    x = Flatten()(model_in.output)

    # xcord path start from mobilenet output
    xcord = Dense(100, activation='relu')(x)
    xcord = Dense(80)(xcord)
    xcord = Dense(1)(xcord)

    # ycord path from mobilenet output
    ycord = Dense(100, activation='relu')(x)
    ycord = Dense(80)(ycord)
    ycord = Dense(1)(ycord)

    model = Model(inputs=model_in.inputs, outputs=[xcord, ycord])

    model.compile(loss = mean_squared_error,
                optimizer = RMSprop(),
                metrics = ['accuracy', 'mae'])

    return model