import rzhang_model
from create_dataset import *
import cPickle

L_SIZE = 224
AB_SIZE = 14

model = rzhang_model.build_model()

# On bhat's google instance
dir_path = '/home/adithya_bhatp/images/nature_hilly/test'
l_channel, one_hot_ab = create_dataset(dir_path, L_SIZE, AB_SIZE)

model.fit(l_channel, one_hot_ab, batch_size = 32, nb_epoch = 32, verbose = 1)

with open('rzhang_model.pickle', 'rb') as fp:
    cPickle.dump(model, fp)
