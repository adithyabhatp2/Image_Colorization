import ImageUtils as myutils
from skimage import color
import os
import numpy as np

BIN_LENGTH = 10
BIN_CENTERS = np.array(range(-105, 106, BIN_LENGTH))
BIN_EDGES = (BIN_CENTERS[:-1] + BIN_CENTERS[1:]) / 2
NUM_BINS = len(BIN_EDGES) + 1

IMG_HEIGHT = 224
IMG_WIDTH = 224



def get_lab(image):
    image_lab = color.rgb2lab(image)

    l = image_lab[:, :, 0]
    ab = image_lab[:, :, 1:]

    return l, ab


def create_dataset(dir_path, l_size, ab_size):
    file_list = os.listdir('dir_path')

    num_images = len(file_list)

    l_channel = np.zeros((num_images, l_size, l_size, 1))
    ab_channels = np.zeros((num_images, ab_size, ab_size, 2))

    for i, f in enumerate(file_list):
        image_path = os.path.join(dir_path, f)
        
        l, ab = get_lab_resized(image_path, l_size, ab_size)

        l_channel[i] = l
        ab_channels[i] = ab

    one_hot = encode_ab(ab_channels)
    return l_channel, one_hot


def get_lab_resized(image_path, l_size, ab_size):
    image = myutils.readColorImageFromFile(image_path)
    l, ab = get_lab(image)

    l = myutils.resizeImage(l, l_size, l_size)
    ab = myutils.resizeImage(ab, ab_size, ab_size)

    return l, ab


# TODO: there is probably a faster way to do this than looping
def encode_ab(ab_channels):
    one_hot = np.zeros((ab_channels.shape[0], ab_channels.shape[1], ab_channels.shape[2], NUM_BINS * NUM_BINS))

    a_bins = np.digitize(ab_channels[:, :, :, 0], BIN_EDGES)
    b_bins = np.digitize(ab_channels[:, :, :, 1], BIN_EDGES)

    ab_bins = a_bins * NUM_BINS + b_bins

    for n in range(ab_channels.shape[0]):
        for i in range(ab_channels.shape[1]):
            for j in range(ab_channels.shape[2]):
                one_hot[n, i, j, ab_bins[n, i, j]] = 1

    return one_hot


def decode_ab(one_hot):
    ab_channels = np.zeros((one_hot.shape[0], one_hot.shape[1], one_hot.shape[2], 2))

    for n in range(one_hot.shape[0]):
        for i in range(one_hot.shape[1]):
            for j in range(one_hot.shape[2]):
                index = np.argmax(one_hot[n, i, j])
                a_index = index / NUM_BINS
                b_index = index % NUM_BINS

                ab_channels[n, i, j, 0] = BIN_CENTERS[a_index]
                ab_channels[n, i, j, 1] = BIN_CENTERS[b_index]



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


if __name__=='__main__':
    TRAIN_DIR = '/u/a/d/adbhat/images/nature_hilly/train_temp'
    VAL_DIR = '/u/a/d/adbhat/images/nature_hilly/val'
    TEST_DIR = '/u/a/d/adbhat/images/nature_hilly/test'

    imagePath = os.path.join(TRAIN_DIR, 'n09246464_86.JPEG')

    original_image = myutils.readColorImageFromFile(imagePath)
    image = myutils.resizeImage(original_image, IMG_HEIGHT, IMG_WIDTH)

    l, ab = get_lab(image)

    print('Min L: {}\nMin A: {}\nMin B: {}'.format(np.amin(l), np.amin(ab[:,:,0]), np.amin(ab[:,:,1])))
    print('Max L: {}\nMax A: {}\nMax B: {}'.format(np.amax(l), np.amax(ab[:, :, 0]), np.amax(ab[:, :, 1])))



    print 'Done'
