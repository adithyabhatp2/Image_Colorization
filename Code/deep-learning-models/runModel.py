from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = '531053453_8ee9337295.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]


# 136310496_216ff74e2f.jpg - ox cart
# 531053453_8ee9337295.jpg - african elephant