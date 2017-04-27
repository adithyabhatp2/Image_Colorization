import ImageUtils as myutils
from skimage import color
import os
import numpy as np

BIN_LENGTH = 10
BIN_CENTERS = np.array(range(-105, 106, BIN_LENGTH))
BIN_EDGES = (BIN_CENTERS[:-1] + BIN_CENTERS[1:]) / 2
NUM_BINS = len(BIN_EDGES) + 1

debugLevel = 2


def get_lab_resized(image_path, l_size, ab_size):
    image = myutils.readColorImageFromFile(image_path)

    image_lab = color.rgb2lab(image)
    l = image_lab[:, :, 0:1]
    ab = image_lab[:, :, 1:]

    l = myutils.resizeImage(l, l_size, l_size)
    ab = myutils.resizeImage(ab, ab_size, ab_size)

    return l, ab


def create_dataset_v2(dir_path, l_size, ab_size):
    file_list = os.listdir(dir_path)
    num_images = len(file_list)

    l_channel = np.zeros((num_images, l_size, l_size, 1))
    ab_channels = np.zeros((num_images, ab_size, ab_size, 2))

    image_index = 0

    for i, f in enumerate(file_list):
        image_path = os.path.join(dir_path, f)
        image = myutils.readColorImageFromFile(image_path)

        # Remove (Some) Grayscale
        # TODO: Can do better. look at channels and remove if max a,b = 0?
        if image.ndim < 3:
            continue

        l, ab = myutils.get_lab(image)

        l = myutils.resizeImage(l, l_size, l_size)
        ab = myutils.resizeImage(ab, ab_size, ab_size)

        l_channel[image_index] = l
        ab_channels[image_index] = ab
        image_index += 1

    l_channel = l_channel[0:image_index, :, :, :]
    ab_channels = ab_channels[0:image_index]

    ab_onehot, ab_bin_frequencies = encode_ab(ab_channels)

    if debugLevel <= 2:
        print('create_datset_v2 : valid 3d images : l_channels shape: {}\nab_channels shape:{}\nab_channels after one-hot shape: {}'.format(l_channel.shape, ab_channels.shape, ab_onehot.shape))

    return l_channel, ab_onehot, ab_bin_frequencies


def create_dataset(dir_path, l_size, ab_size):
    file_list = os.listdir(dir_path)

    num_images = len(file_list)

    l_channel = np.zeros((num_images, l_size, l_size, 1))
    ab_channels = np.zeros((num_images, ab_size, ab_size, 2))

    for i, f in enumerate(file_list):
        image_path = os.path.join(dir_path, f)

        l, ab = get_lab_resized(image_path, l_size, ab_size)

        l_channel[i] = l
        ab_channels[i] = ab

    one_hot, _ = encode_ab(ab_channels)
    return l_channel, one_hot


# TODO: there is probably a faster way to do this than looping
def encode_ab(ab_channels):
    num_images = ab_channels.shape[0]
    img_height = ab_channels.shape[1]
    img_width = ab_channels.shape[2]

    one_hot = np.zeros((num_images, img_height, img_width, NUM_BINS * NUM_BINS))

    ab_bin_freqs = np.zeros(NUM_BINS * NUM_BINS, dtype=np.float64)

    a_bins = np.digitize(ab_channels[:, :, :, 0], BIN_EDGES)
    b_bins = np.digitize(ab_channels[:, :, :, 1], BIN_EDGES)

    ab_bins = a_bins * NUM_BINS + b_bins

    for n in range(num_images):
        for i in range(img_height):
            for j in range(img_width):
                one_hot[n, i, j, ab_bins[n, i, j]] = 1
                ab_bin_freqs[ab_bins[n, i, j]] += 1

    return one_hot, ab_bin_freqs


def decode_ab(n_one_hots):
    ab_channels = np.zeros((n_one_hots.shape[0], n_one_hots.shape[1], n_one_hots.shape[2], 2))

    for n in range(n_one_hots.shape[0]):
        for i in range(n_one_hots.shape[1]):
            for j in range(n_one_hots.shape[2]):
                index = np.argmax(n_one_hots[n, i, j])
                a_index = index / NUM_BINS
                b_index = index % NUM_BINS

                ab_channels[n, i, j, 0] = BIN_CENTERS[a_index]
                ab_channels[n, i, j, 1] = BIN_CENTERS[b_index]

    return ab_channels


def checkLabLimitsFloat():
    # 000, 001, 010, 011
    # 100, 101, 110, 111
    # Ref http://colorizer.org/
    r = np.ones((16, 16))
    g = np.zeros((16, 16))
    b = np.ones((16, 16))
    temp = np.array(zip(r, g, b))
    print temp.shape
    rgb = np.swapaxes(temp, 1, 2)
    print rgb.shape

    print('RGB:\n{}'.format(rgb[0, :2]))
    lab = color.rgb2lab(rgb)
    print('LAB:\n{}'.format(lab[0, :2]))
    rgb1 = color.lab2rgb(lab)
    print('RGB1:\n{}'.format(rgb[0, :2]))

    # l extremes 0 to +100 - 000, 111
    # a extreme -86 to +98 - 010, 101
    # b extreme val -108 to 95 - 011


if __name__ == '__main__':
    # TRAIN_DIR = '/u/a/d/adbhat/images/nature_hilly/train_temp'
    # VAL_DIR = '/u/a/d/adbhat/images/nature_hilly/val'
    # TEST_DIR = '/u/a/d/adbhat/images/nature_hilly/test'

    # imagePath = os.path.join(TRAIN_DIR, 'n09246464_86.JPEG')
    #
    # original_image = myutils.readColorImageFromFile(imagePath)
    # image = myutils.resizeImage(original_image, IMG_HEIGHT, IMG_WIDTH)
    #
    # l, ab = myutils.get_lab(image)
    #
    # print('Min L: {}\nMin A: {}\nMin B: {}'.format(np.amin(l), np.amin(ab[:,:,0]), np.amin(ab[:,:,1])))
    # print('Max L: {}\nMax A: {}\nMax B: {}'.format(np.amax(l), np.amax(ab[:, :, 0]), np.amax(ab[:, :, 1])))


    TRAIN_DIR = '/home/adithya_bhatp/images/nature_hilly/train'
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    AB_SIZE = 56

    l_channels, ab_channels_onehot, ab_bin_frequencies = create_dataset_v2(TRAIN_DIR, IMG_HEIGHT, AB_SIZE)

    # # Test Reconstruction
    # ab_channels = decode_ab(ab_channels_onehot)
    # reconstructed_rgb_imgs = []
    # for i in xrange(0,len(l_channels)):
    #     reconstructed_lab_img = myutils.merge_l_ab(l_channels[i], ab_channels[i])
    #     rgb_img = color.lab2rgb(reconstructed_lab_img)
    #     reconstructed_rgb_imgs.append(rgb_img)
    # myutils.displayListOfImagesInGrid(reconstructed_rgb_imgs)

    # Test ab_bin_frequencies
    print('A-B bin frequencies\nLength: {}\tNon-Zero: {}\nSum: {}\nExpected: {}\t'.format(len(ab_bin_frequencies), np.count_nonzero(ab_bin_frequencies), sum(ab_bin_frequencies), 8 * AB_SIZE * AB_SIZE))
    print('Counts: \n{}'.format(ab_bin_frequencies))

    print 'Done'
