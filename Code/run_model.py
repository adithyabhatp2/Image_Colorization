import rzhang_model
import ImageUtils as myutils
from create_dataset import *
import cPickle
import time

L_SIZE = 224
AB_SIZE = 14

model = rzhang_model.build_model()

# On bhat's google instance
dir_path = '/home/adithya_bhatp/images/nature_hilly/train'

start_time = time.time()
# myutils.cleanup_dir(dir_path)

l_channel, one_hot_ab = create_dataset(dir_path, L_SIZE, AB_SIZE)

stop_time = time.time()
print 'Parsing time: {0}'.format(stop_time - start_time)

start_time = time.time()
model.fit(l_channel, one_hot_ab, batch_size = 32, epochs = 1, verbose = 1)
stop_time = time.time()
print 'Model fit time: {0}'.format(stop_time - start_time)

model.save('rzhang_model.h5')

