from skimage.io import imread
from skimage.viewer import ImageViewer

import skimage
import skimage.transform
import skimage.color
import math

import matplotlib.pyplot as plt
import numpy as np


def resizeImage(image, new_height, new_width):
    # shape returns height, width, channels
    resized = skimage.transform.resize(image=image, output_shape=(new_height, new_width, image.shape[2]))
    return resized


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

# variable number of images
# If pairUp=True, adjacent images need to be of same size
def displayImagesInGrid(pairUp=False, *images):
    numImages = len(images)
    numRows = int(math.ceil(math.sqrt(numImages)))
    numCols = int(math.ceil(numImages / numRows))
    if numCols % 2 != 0:
        numCols += 1
    if pairUp:
        numCols /= 2

    print("Num images: {}\trows{}\tcols{}".format(numImages, numRows, numCols))

    base_fig = plt.figure()
    subplotNum = 0

    for i in xrange(0, numRows):
        for j in xrange(0, numCols):
            subplotNum += 1
            img_index = (i * numRows) + j
            if img_index >= numImages:
                break
            img = images[img_index]
            if pairUp:
                img_index += 1
                j += 1
                img2 = images[img_index]
                img = np.hstack((img, img2))
            subplot = base_fig.add_subplot(numRows, numCols, subplotNum)
            subplot.set_title(img_index)
            imgplot = plt.imshow(img)

    plt.show()


if __name__ == '__main__':
    imagePath = '/u/a/d/adbhat/private/gitRepository/images/n00433661_outdoorsport_1351/n00433661_7.JPEG'
    image = readColorImageFromFile(imagePath)

    resized_image = resizeImage(image, 224, 224)
    grey_img = convertImageToGray2D(resized_image)
    grey_rgb_img = convertImageToGrayRGB(resized_image)

    print('Resized shape:{}\tGreyed shape:{}\tGrey_RGB shape:{}'.format(resized_image.shape, grey_img.shape, grey_rgb_img.shape))

    # displayImage(image)
    # displayImage(resized_image)
    # displayImage(grey_img)
    # displayImage(grey_rgb_img)

    imageUrl = 'https://i.ytimg.com/vi/g55evwFJWBw/maxresdefault.jpg'
    # displayImage(readColorImageFromUrl(imageUrl))
    # displayImage(convertImageToGray(readColorImageFromUrl(imageUrl)))

    # displayImagesInGrid(False, image, resized_image, grey_img, grey_rgb_img, image)
    displayImagesInGrid(True, resized_image, grey_rgb_img, grey_rgb_img, resized_image)
