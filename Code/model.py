import cPickle
import numpy as np
from copy import copy, deepcopy

from skimage import color
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Flatten, Dense, Reshape

nb_epoch = 5

with open('../cifar100/cifar100_large_natural_outdoor_scenes.pickle', 'rb') as fp:
    data = cPickle.load(fp)
    X_train, X_test = data

l_factor = 128.0
a_factor = 256.0
b_factor = 256.0

X_train[:, :, :, 0] /= l_factor
X_train[:, :, :, 1] /= a_factor
X_train[:, :, :, 2] /= b_factor

X_test[:, :, :, 0] /= l_factor
X_test[:, :, :, 1] /= a_factor
X_test[:, :, :, 2] /= b_factor


model_a = Sequential()

model_a.add(Convolution2D(64, 3, 3, input_shape = (32, 32, 1)))
model_a.add(BatchNormalization())
model_a.add(Activation('relu'))
model_a.add(MaxPooling2D(pool_size = (2, 2)))

model_a.add(Convolution2D(128, 3, 3))
model_a.add(BatchNormalization())
model_a.add(Activation('relu'))
model_a.add(MaxPooling2D(pool_size = (2, 2)))

model_a.add(Flatten())

model_a.add(Dense(1024, activation = 'tanh'))
model_a.add(Reshape((32, 32)))

model_a.compile(loss = 'mean_squared_error', optimizer = 'adam')
model_a.fit(X_train[:, :, :, 0:1], X_train[:, :, :, 1], batch_size = 32, nb_epoch = nb_epoch, verbose = 1)

a_train_est = model_a.predict(X_train[:, :, :, 0:1])
a_test_est = model_a.predict(X_test[:, :, :, 0:1])


model_b = Sequential()

model_b.add(Convolution2D(64, 3, 3, input_shape = (32, 32, 1)))
model_b.add(BatchNormalization())
model_b.add(Activation('relu'))
model_b.add(MaxPooling2D(pool_size = (2, 2)))

model_b.add(Convolution2D(128, 3, 3))
model_b.add(BatchNormalization())
model_b.add(Activation('relu'))
model_b.add(MaxPooling2D(pool_size = (2, 2)))

model_b.add(Flatten())

model_b.add(Dense(1024, activation = 'tanh'))
model_b.add(Reshape((32, 32)))

model_b.compile(loss = 'mean_squared_error', optimizer = 'adam')
model_b.fit(X_train[:, :, :, 0:1], X_train[:, :, :, 2], batch_size = 32, nb_epoch = nb_epoch, verbose = 1)

b_train_est = model_b.predict(X_train[:, :, :, 0:1])
b_test_est = model_b.predict(X_test[:, :, :, 0:1])


X_train_est = deepcopy(X_train)
X_train_est[:, :, :, 1] = a_train_est
X_train_est[:, :, :, 2] = b_train_est

X_test_est = deepcopy(X_test)
X_test_est[:, :, :, 1] = a_test_est
X_test_est[:, :, :, 2] = b_test_est

X_train[:, :, :, 0] *= l_factor
X_train[:, :, :, 1] *= a_factor
X_train[:, :, :, 2] *= b_factor

X_test[:, :, :, 0] *= l_factor
X_test[:, :, :, 1] *= a_factor
X_test[:, :, :, 2] *= b_factor

X_train_est[:, :, :, 0] *= l_factor
X_train_est[:, :, :, 1] *= a_factor
X_train_est[:, :, :, 2] *= b_factor

X_test_est[:, :, :, 0] *= l_factor
X_test_est[:, :, :, 1] *= a_factor
X_test_est[:, :, :, 2] *= b_factor


def show_plots(i, actual, estimate):
    plt.imshow(color.lab2rgb(actual[i]))
    plt.show(block = False)
    plt.figure()
    plt.imshow(color.lab2rgb(estimate[i]))
    plt.show(block = False)
    plt.figure()
    plt.imshow(color.rgb2gray(color.lab2rgb(actual[i])), cmap = 'gray')
    plt.show(block = False)

# show_plots(500, X_test, X_test_est)
# show_plots(50, X_train, X_train_est)


import random
def show_grid(dataset, gray = False):
    image_col_list = []
    for i in range(10):
        image_row_tuple = tuple(dataset[i*10 + j] for j in range(10))
        I = np.concatenate(image_row_tuple, 1)
        image_col_list.append(I)
    image_col_tuple = tuple(image_col_list)
    I = np.concatenate(image_col_tuple, 0)
    I = color.lab2rgb(I)
    plt.figure()
    if gray:
        I = color.rgb2gray(I)
        plt.imshow(I, cmap = 'gray')
    else:
        plt.imshow(I)
    plt.show(block = False)

show_grid(X_test, gray = True)
show_grid(X_test_est)
show_grid(X_train, gray = True)
show_grid(X_train_est)

show_grid(X_test)
show_grid(X_test_est)
show_grid(X_train)
show_grid(X_train_est)
