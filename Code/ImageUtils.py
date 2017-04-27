from skimage.io import imread
from skimage.viewer import ImageViewer

import skimage
import skimage.transform
import skimage.color
import math

import matplotlib.pyplot as plt
import numpy as np

import os

def cleanup_dir(dir_path):
    file_list = os.listdir(dir_path)

    for f in file_list:
        image_path = os.path.join(dir_path, f)
        image = readColorImageFromFile(image_path)

        if image.ndim != 3:
            os.system('rm -f {0}'.format(image_path))


def resizeImage(image, new_height, new_width):
    """
    Assumes channels_last keras encoding - tensorflow default.
    :param image:
    :param new_height:
    :param new_width:
    :return: resized image with the same number of channels.
    """
    # shape returns height, width, channels
    if image.ndim == 2:
        resized = skimage.transform.resize(image=image, output_shape=(new_height, new_width))
    else:
        resized = skimage.transform.resize(image=image, output_shape=(new_height, new_width, image.shape[2]))
    return resized


def get_lab(image):
    image_lab = skimage.color.rgb2lab(image)
    l = image_lab[:, :, 0:1]
    ab = image_lab[:, :, 1:]
    return l, ab


# https://github.com/scikit-image/scikit-image/issues/1201 Could throw warnings about color data range
def merge_l_ab(l, ab):
    ab_resized = resizeImage(ab, l.shape[0], l.shape[1])
    merged = np.concatenate((l, ab_resized), axis=2)
    return merged


def displayImage(image):
    viewer = ImageViewer(image)
    viewer.show()


def readColorImageFromFile(filePath, as_grey=False):
    return skimage.img_as_float(imread(fname=filePath, as_grey=as_grey))


def readColorImageFromUrl(url, as_grey=False):
    return skimage.img_as_float(imread(url, as_grey=as_grey))


# 2d float array with range 0 to 1
def convertImageToGray2D(image):
    return skimage.color.rgb2grey(image)


def convertImageToGrayRGB(image):
    return skimage.color.gray2rgb(skimage.color.rgb2grey(image))


def convertLabToRgb(image):
    return skimage.color.lab2rgb(image)

def displayListOfImagesInGrid(imageList, pairUp=False):
    numImages = len(imageList)
    numCols = int(math.ceil(math.sqrt(numImages)))
    if numCols % 2 != 0:
        numCols += 1
    numRows = int(math.ceil(numImages / numCols))

    if pairUp:
        numCols /= 2

    print("Num images: {}\trows{}\tcols{}".format(numImages, numRows, numCols))

    base_fig = plt.figure()
    subplotNum = 0

    for i in xrange(0, numRows):
        for j in xrange(0, numCols):

            subplotNum += 1
            img_index = subplotNum - 1
            # TODO:pairUp is prolly messed up.

            if img_index >= numImages:
                break
            img = imageList[img_index]

            if pairUp:
                img_index += 1
                j += 1
                img2 = imageList[img_index]
                img = np.hstack((img, img2))

            subplot = base_fig.add_subplot(numRows, numCols, subplotNum)
            subplot.set_title(img_index)
            imgplot = plt.imshow(img)

    plt.show()

def get_ab_bin_weights():
    AB_BIN_FREQS = [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   3.00000000e+00,   1.00000000e+00,   1.00000000e+00,   2.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   8.00000000e+00,   1.20000000e+01,   5.00000000e+00,   4.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   4.00000000e+00,   3.40000000e+01,   4.30000000e+01,   7.00000000e+00,   4.50000000e+01,   7.20000000e+01,   1.43000000e+02,   1.41000000e+02,   5.10000000e+01,   1.50000000e+01,   2.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.10000000e+01,   5.70000000e+01,   9.80000000e+01,   8.20000000e+01,   1.84000000e+02,   4.89000000e+02,   1.16600000e+03,   1.16200000e+03,   6.26000000e+02,   1.93000000e+02,   3.30000000e+01,   4.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.00000000e+00,   1.17000000e+02,   5.45000000e+02,   1.13900000e+03,   9.58000000e+02,   9.85000000e+02,   2.06100000e+03,   5.11800000e+03,   7.10700000e+03,   4.55700000e+03,   1.41500000e+03,   3.35000000e+02,   5.90000000e+01,   1.10000000e+01,   3.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.66000000e+02,   1.70300000e+03,   5.15900000e+03,   7.87300000e+03,   1.02690000e+04,   1.20980000e+04,   2.26550000e+04,   2.42960000e+04,   1.49030000e+04,   5.34000000e+03,   1.41100000e+03,   3.85000000e+02,   1.49000000e+02,   4.60000000e+01,   2.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.35000000e+02,   2.80200000e+03,   1.63060000e+04,   4.06660000e+04,   7.50860000e+04,   1.47989000e+05,   2.09351000e+05,   6.98960000e+04,   3.00650000e+04,   1.03820000e+04,   3.09900000e+03,   8.73000000e+02,   2.33000000e+02,   8.10000000e+01,   2.40000000e+01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   8.80000000e+01,   1.97800000e+03,   9.55500000e+03,   1.87210000e+04,   2.46440000e+04,   4.02270000e+04,   1.14649000e+05,   1.09503000e+05,   6.42830000e+04,   2.80300000e+04,   8.69000000e+03,   2.51200000e+03,   7.63000000e+02,   1.46000000e+02,   4.70000000e+01,   1.50000000e+01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   7.00000000e+00,   5.22000000e+02,   2.63900000e+03,   3.87200000e+03,   2.93700000e+03,   2.06000000e+03,   1.77200000e+03,   2.89300000e+03,   7.05900000e+03,   1.26010000e+04,   1.22930000e+04,   5.71100000e+03,   2.03600000e+03,   6.40000000e+02,   2.17000000e+02,   9.10000000e+01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.05000000e+02,   5.88000000e+02,   8.74000000e+02,   7.72000000e+02,   3.33000000e+02,   1.14000000e+02,   1.78000000e+02,   2.64000000e+02,   7.31000000e+02,   2.00100000e+03,   3.49800000e+03,   2.88100000e+03,   1.28900000e+03,   4.29000000e+02,   1.14000000e+02,   2.70000000e+01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.90000000e+01,   1.99000000e+02,   2.49000000e+02,   1.89000000e+02,   6.20000000e+01,   2.10000000e+01,   1.70000000e+01,   3.40000000e+01,   5.50000000e+01,   1.13000000e+02,   2.76000000e+02,   6.77000000e+02,   1.13300000e+03,   9.99000000e+02,   3.84000000e+02,   8.70000000e+01,   1.20000000e+01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   4.80000000e+01,   6.40000000e+01,   3.60000000e+01,   9.00000000e+00,   3.00000000e+00,   8.00000000e+00,   1.70000000e+01,   9.00000000e+00,   1.90000000e+01,   3.00000000e+01,   1.11000000e+02,   1.26000000e+02,   2.06000000e+02,   1.90000000e+02,   2.18000000e+02,   7.20000000e+01,   1.10000000e+01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   7.00000000e+00,   1.30000000e+01,   5.00000000e+00,   0.00000000e+00,   2.00000000e+00,   5.00000000e+00,   7.00000000e+00,   1.00000000e+01,   5.00000000e+00,   6.00000000e+00,   1.90000000e+01,   6.40000000e+01,   7.40000000e+01,   6.00000000e+01,   4.10000000e+01,   1.40000000e+01,   2.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.00000000e+00,   2.00000000e+00,   1.00000000e+00,   0.00000000e+00,   2.00000000e+00,   3.00000000e+00,   4.00000000e+00,   8.00000000e+00,   2.90000000e+01,   2.70000000e+01,   3.30000000e+01,   6.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.00000000e+00,   6.00000000e+00,   2.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   6.00000000e+00,   1.10000000e+01,   6.00000000e+00,   1.00000000e+01,   3.30000000e+01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.00000000e+00,   0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00  ]
    # AB_BIN_FREQS = [c + 1 for c in AB_BIN_FREQS]

    AB_BIN_FREQS = [float(c) / sum(AB_BIN_FREQS) for c in AB_BIN_FREQS]

    AB_BIN_WEIGHTS = [0.5 * c + 0.5 / len(AB_BIN_FREQS) for c in AB_BIN_FREQS]
    AB_BIN_WEIGHTS = [1.0 / w for w in AB_BIN_WEIGHTS]

    AB_BIN_WEIGHTS = [w / sum(AB_BIN_WEIGHTS) for w in AB_BIN_WEIGHTS]

    print AB_BIN_WEIGHTS



if __name__ == '__main__':
    # imagePath = '/u/a/d/adbhat/images/nature_hilly/train_temp/n09246464_86.JPEG'
    # image = readColorImageFromFile(imagePath)
    #
    # resized_image = resizeImage(image, 224, 224)
    # grey_img = convertImageToGray2D(resized_image)
    # grey_rgb_img = convertImageToGrayRGB(resized_image)
    #
    # print('Resized shape:{}\tGreyed shape:{}\tGrey_RGB shape:{}'.format(resized_image.shape, grey_img.shape, grey_rgb_img.shape))

    # displayImage(image)
    # displayImage(resized_image)
    # displayImage(grey_img)
    # displayImage(grey_rgb_img)

    imageUrl = 'https://i.ytimg.com/vi/g55evwFJWBw/maxresdefault.jpg'
    # displayImage(readColorImageFromUrl(imageUrl))
    # displayImage(convertImageToGray(readColorImageFromUrl(imageUrl)))

    # displayImagesInGrid(False, image, resized_image, grey_img, grey_rgb_img, image)
    # displayImagesInGrid(True, resized_image, grey_rgb_img, grey_rgb_img, resized_image)
