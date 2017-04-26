import keras
from keras.activations import softmax
import rzhang_model
import ImageUtils as myutils
from create_dataset import *
import cPickle
import time
import h5py

keras.activations.softmax3 = rzhang_model.softmax3
keras.losses.custom_loss = rzhang_model.custom_loss

L_SIZE = 224
AB_SIZE = 14


# On bhat's google instance
dir_path = '/home/adithya_bhatp/images/nature_hilly/val'

start_time = time.time()

# myutils.cleanup_dir(dir_path)
l_channel, one_hot_ab = create_dataset(dir_path, L_SIZE, AB_SIZE)

stop_time = time.time()
print 'Parsing time: {0}'.format(stop_time - start_time)

for i in range(6):
    try:
        f = h5py.File('rzhang_model{0}.h5'.format(i), 'r+')
        del f['optimizer_weights']
        f.close()
    except KeyError:
        pass

    model = keras.models.load_model('rzhang_model{0}.h5'.format(i))
    loss = model.evaluate(l_channel, one_hot_ab)

    print i, loss

