from skimage.io import imread
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import skimage
import skimage.transform
import skimage.color


def resizeImage(image, new_height, new_width):
    # shape returns height, width, channels
    resized = skimage.transform.resize(image=image, output_shape=(new_height, new_width, image.shape[2]))
    return resized

def displayImage(image):
    viewer = ImageViewer(image)
    viewer.show()

def readColorImageFromFile(filePath):
    return imread(fname=imagePath)

def convertImageToGray(image):
    return skimage.color.rgb2grey(image)

def convertImageToGray(image):
    return skimage.color.rgb2grey(image)

if __name__ == '__main__':
    imagePath = '/u/a/d/adbhat/private/gitRepository/images/n00433661_outdoorsport_1351/n00433661_7.JPEG'
    image = imread(fname=imagePath)
    print image.shape
    reshaped_image = resizeImage(image, 224, 224)
