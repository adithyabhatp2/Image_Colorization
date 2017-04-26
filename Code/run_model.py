import keras
import rzhang_model
import ImageUtils as myutils
from create_dataset import *
import cPickle

keras.activations.softmax3 = rzhang_model.softmax3
keras.losses.custom_loss = rzhang_model.custom_loss

L_SIZE = 224
AB_SIZE = 14

# model = rzhang_model.build_model()
model = keras.models.load_model('rzhang_model5.h5')

# On bhat's google instance
dir_path = '/home/adithya_bhatp/images/nature_hilly/train'

# myutils.cleanup_dir(dir_path)

l_channel, one_hot_ab = create_dataset(dir_path, L_SIZE, AB_SIZE)

for i in range(10):
    model.fit(l_channel, one_hot_ab, batch_size = 32, epochs = 5, verbose = 1)
    model.save('rzhang_model{0}.h5'.format(i + 6))

