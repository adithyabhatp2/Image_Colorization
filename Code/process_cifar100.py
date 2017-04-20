from keras.datasets import cifar100
import matplotlib.pyplot as plt
import cPickle
from skimage import color
import numpy as np
import cPickle

with open('../cifar100/meta', 'rb') as fmeta:
    meta = cPickle.load(fmeta)
    labels = meta['coarse_label_names']

(X_train, y_train), (X_test, y_test) = cifar100.load_data('coarse')
y_train = y_train.flatten()
y_test = y_test.flatten()

large_natural_outdoor_scenes_label = labels.index('large_natural_outdoor_scenes')

X_train = X_train[y_train == large_natural_outdoor_scenes_label]
X_test = X_test[y_test == large_natural_outdoor_scenes_label]

del y_train, y_test

X_train_lab = np.zeros(X_train.shape)
X_test_lab = np.zeros(X_test.shape)

for i in range(X_train.shape[0]):
    X_train_lab[i] = color.rgb2lab(X_train[i])

for i in range(X_test.shape[0]):
    X_test_lab[i] = color.rgb2lab(X_test[i])

X_train = X_train_lab
X_test = X_test_lab

data = (X_train, X_test)
with open('../cifar100/cifar100_large_natural_outdoor_scenes.pickle', 'wb') as fp:
    cPickle.dump(data, fp)
