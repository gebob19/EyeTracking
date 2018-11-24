import tensorflow as tf
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model
# from keras.losses import mean_squared_error
# from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Reshape

# try pretrained weights and default config to get a baseline => (224, 224, 3) inputs
# shape = (224, 224, 3)
# mobnet hyper param
alpha = 1
depth_multiplier = 1

ycam_alpha = 1
ycam_dm = 1

xcam_alpha = 1
xcam_dm = 1

# reshape shape for gateway mobnet output
def mobnet(shape, weights):
    model_in = MobileNetV2(input_shape=shape,
                    include_top=False,
                    weights=weights,
                    pooling=None)

    x = Flatten()(model_in.output)
    x = Dense(100, activation=None)(x)
    y = Dense(2, activation=None)(x)

    model = Model(inputs=model_in.input, outputs=y)

    return model