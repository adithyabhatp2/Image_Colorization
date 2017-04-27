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
