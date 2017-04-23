import keras
import keras.backend as K

from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Flatten, Dense, Reshape

from keras.activations import softmax

import functools
softmax3 = functools.partial(softmax, axis = 3)


def custom_loss(y_true, y_pred):
    return K.sum(y_true * K.log(y_pred))


# TODO: upgrade keras using:
# sudo pip install git+git://github.com/fchollet/keras.git --upgrade


def build_model():
    model = Sequential()

    #### conv1
    # conv1_1
    model.add(ZeroPadding2D(padding = (1, 1), input_shape = (224, 224, 1)))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))

    # conv1_2
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(64, (3, 3), strides = (2, 2)))
    model.add(Activation('relu'))

    # conv1_2norm
    model.add(BatchNormalization())


    #### conv2
    # conv2_1
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(128, (3, 3)))
    model.add(Activation('relu'))

    # conv2_2
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(128, (3, 3), strides = (2, 2)))
    model.add(Activation('relu'))

    # conv2_2norm
    model.add(BatchNormalization())


    #### conv3
    # conv3_1
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, (3, 3)))
    model.add(Activation('relu'))

    # conv3_2
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, (3, 3)))
    model.add(Activation('relu'))

    # conv3_3
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, (3, 3), strides = (2, 2)))
    model.add(Activation('relu'))

    # conv3_3norm
    model.add(BatchNormalization())


    #### conv4
    # conv4_1
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(Activation('relu'))

    # conv4_2
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(Activation('relu'))

    # conv4_3
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(Activation('relu'))

    #conv4_3norm
    model.add(BatchNormalization())


    #### conv5
    # conv5_1
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(512, (3, 3), dilation_rate = (2, 2)))
    model.add(Activation('relu'))

    # conv5_2
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(512, (3, 3), dilation_rate = (2, 2)))
    model.add(Activation('relu'))

    # conv5_3
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(512, (3, 3), dilation_rate = (2, 2)))
    model.add(Activation('relu'))

    # conv5_3norm
    model.add(BatchNormalization())


    #### conv6
    # conv6_1
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(512, (3, 3), dilation_rate = (2, 2)))
    model.add(Activation('relu'))

    # conv6_2
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(512, (3, 3), dilation_rate = (2, 2)))
    model.add(Activation('relu'))

    # conv6_3
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(512, (3, 3), dilation_rate = (2, 2)))
    model.add(Activation('relu'))

    # conv6_3norm
    model.add(BatchNormalization())


    #### conv7
    # conv7_1
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(Activation('relu'))

    # conv7_2
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(Activation('relu'))

    # conv7_3
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(Activation('relu'))

    # conv7_3norm
    model.add(BatchNormalization())


    #### conv8
    # conv8_1
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, (4, 4), strides = (2, 2)))
    model.add(Activation('relu'))

    # conv8_2
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, (3, 3)))
    model.add(Activation('relu'))

    # conv8_3
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, (3, 3)))
    model.add(Activation('relu'))


    #### Softmax
    model.add(Convolution2D(313, (1, 1)))
    model.add(Activation(softmax3))

    ## Loss
    model.compile(loss = custom_loss, optimizer = 'adam')

    return model


model = build_model()
print model.output_shape
