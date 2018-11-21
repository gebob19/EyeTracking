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
nshape = (224, 280, 1)

def mobnet(shape):
    model_in = MobileNetV2(input_shape=shape,
#                     alpha=alpha,
#                     depth_multiplier=depth_multiplier,
                    include_top=False,
                    weights='imagenet',
                    pooling=None)
    rename_layers(model_in, 'gateway')
    x = Reshape(nshape)(model_in.output)

    # prediction of x gaze position
    xcam_net = MobileNetV2(input_tensor=x,
                        alpha=xcam_alpha,
                        depth_multiplier=xcam_dm,
                        include_top=False,
                        weights=None)
    xcam_out = Flatten()(xcam_net.output)
    y_xcam = Dense(1, activation=None, name='y_xcam')(xcam_out)

    # prediction of y gaze position
    ycam_net = MobileNetV2(input_tensor=x,
                        alpha=ycam_alpha,
                        depth_multiplier=ycam_dm,
                        include_top=False,
                        weights=None)
    ycam_out = Flatten()(ycam_net.output)
    y_ycam = Dense(1, activation=None, name='y_ycam')(ycam_out)

    # fix to keras multiple layers with same name bug
    rename_layers(xcam_net, '-xgaze')
    rename_layers(ycam_net, '-ygaze')

    model = Model(inputs=model_in.input, outputs=[y_xcam, y_ycam])

    return model


def rename_layers(model, postfix):
    for l in model.layers: l.name = l.name + postfix