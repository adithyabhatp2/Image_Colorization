import ImageUtils as imgUtils
import create_dataset as datasetUtils
import numpy as np
import keras.models
import keras.layers
import rzhang_model
import time



# http://stackoverflow.com/questions/11765061/better-way-to-shuffle-two-related-lists
def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def test_shuffle_unison():
    a = range(1,10)
    b = range(11,20)
    c = range(21,30)

    d = zip(b,c)

    for i in range(1,6):
        shuffle_in_unison_scary(a,d)
        print('a : {}\nb: {}'.format(a,d))


if __name__ == '__main__':
    TRAIN_DIR = '/u/a/d/adbhat/images/nature_hilly/train'
    VAL_DIR = '/u/a/d/adbhat/images/nature_hilly/val'
    TEST_DIR = '/u/a/d/adbhat/images/nature_hilly/test'

    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    AB_SIZE = 24

    # Load or intialize Model
    keras.activations.softmax3 = rzhang_model.softmax3
    keras.losses.custom_loss = rzhang_model.custom_loss
    model = keras.models.load_model('rzhang_model5.h5')

    # Load Training Data
    load_train_start_time = time.time()
    train_ls, train_abs_onehot, train_ab_bin_frequencies = datasetUtils.create_dataset_v2(TRAIN_DIR, IMG_HEIGHT, IMG_HEIGHT)
    load_train_end_time = time.time()

    print('Num train examples: {}'.format(len(train_ls)))
    print('Load time: {}'.format(load_train_end_time-load_train_start_time))
    print('Train ab bin frequencies: \n{}'.format(train_ab_bin_frequencies))

    # Train
    NUM_EPOCHS = 10
    for epoch in xrange(0,NUM_EPOCHS):
        history = model.fit(x=train_ls, y=train_abs_onehot, epochs=1, initial_epoch=epoch, shuffle=False)
        # Can also give validation data here - not sure how to plot it out though
        # Need to shuffle manually, since we want to dump out the model after each epoch




