import ImageUtils as myutils
from create_dataset import *
from skimage import color
import os
import numpy as np

BIN_LENGTH = 1
BIN_CENTERS = np.array(range(-105, 106, BIN_LENGTH))
BIN_EDGES = (BIN_CENTERS[:-1] + BIN_CENTERS[1:]) / 2
NUM_BINS = len(BIN_EDGES) + 1

if __name__ == '__main__':

    L_SIZE = 224
    AB_SIZE = 28
    ab_size = AB_SIZE

    # On bhat's google instance
    train_dir_path = '/home/adithya_bhatp/images/nature_hilly/train'
    val_dir_path = '/home/adithya_bhatp/images/nature_hilly/val'
    test_dir_path = '/home/adithya_bhatp/images/nature_hilly/test'
    test_dir_path = '/home/adithya_bhatp/images/nature_hilly/test_sample' # Make a sample dir with 10 imgs or something

    ab_bin_freqs = np.zeros(NUM_BINS * NUM_BINS, dtype=np.float32)
    

    # train_l_channel, train_one_hot_ab, train_ab_bin_frequencies = create_dataset_v2(train_dir_path, L_SIZE, AB_SIZE)

    file_list = os.listdir(train_dir_path)
    num_images = len(file_list)
    
    batch_size=50
    for batch_num in range(num_images/batch_size):
        print('Batch id: {} of {} '.format(batch_num, (num_images/batch_size)))
        ab_channels = np.zeros((batch_size, ab_size, ab_size, 2))
        image_index = 0
        for i in range(batch_size*(batch_num), batch_size*(batch_num+1)):
            image_path = os.path.join(train_dir_path, file_list[i])
            image = myutils.readColorImageFromFile(image_path)

            if image.ndim < 3:
                continue

            _, ab = myutils.get_lab(image)
            ab = myutils.resizeImage(ab, ab_size, ab_size)

            ab_channels[image_index] = ab 
            image_index += 1

        ab_channels = ab_channels[0:image_index]

        img_height = ab_channels.shape[1]
        img_width = ab_channels.shape[2]

        print('One-hot conversion - training data')

        one_hot = np.zeros((batch_size, img_height, img_width, NUM_BINS * NUM_BINS), dtype = np.bool)

        a_bins = np.digitize(ab_channels[:, :, :, 0], BIN_EDGES)
        b_bins = np.digitize(ab_channels[:, :, :, 1], BIN_EDGES)

        ab_bins = a_bins * NUM_BINS + b_bins

        print('AB binning / counting - training data')
        for n in range(batch_size):
            for i in range(img_height):
                for j in range(img_width):
                    one_hot[n, i, j, ab_bins[n, i, j]] = 1
                    ab_bin_freqs[ab_bins[n, i, j]] += 1


    np.savetxt("ab_freqs_1.txt", ab_bin_freqs)
    print ab_bin_freqs


    