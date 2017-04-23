import ImageUtils as myutils
from skimage import color
from os import listdir

BIN_LENGTH = 10
BIN_CENTERS = np.array(range(-105, 106, BIN_LENGTH))
BIN_EDGES = (BIN_CENTERS[:-1] + BIN_CENTERS[1:]) / 2
NUM_BINS = len(BIN_EDGES) + 1

def get_lab(image_path, l_size, ab_size):
    image = myutils.readColorImageFromFile(image_path)
    image_lab = color.rgb2lab(image)

    l = resizeImage(image, l_size, l_size)
    ab = resizeImage(image[:, :, 1:], ab_size, ab_size)

    return l, ab

# /home/adithya_bhatp/images/n00467719_sport_athleticgame_outdoorgame_fieldgame_1430/n00467719_7006.JPEG
# l_size > ab_size
def create_dataset(dir_path, l_size, ab_size):
    file_list = listdir('dir_path')

    num_images = len(file_list)
    
    l_channel = np.zeros((num_images, l_size, l_size, 1))
    ab_channels = np.zeros((num_images, ab_size, ab_size, 2))

    for i, f in enumerate(file_list):
        image_path = dir_path + "/" + f
        l, ab = get_lab(image_path, l_size, ab_size)

        l_channel[i] = l
        ab_channels[i] = ab

    return l_channel, ab_channels


# TODO: there is probably a faster way to do this instead of looping
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
