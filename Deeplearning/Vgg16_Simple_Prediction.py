import keras
from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
#checking the version of Keras, in my system it is: 1.2
print keras.__version__

# downloading the VGG16 model with transferlearning 
vgg = vgg16.VGG16(weights='imagenet', include_top=True)

# loading image from classpath and converting image to array
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)

#reshaping the array
x = np.expand_dims(x, axis=0)
performing standard preporcessing given byt the vg166 model itself
x = preprocess_input(x)

#predicting the top3 categories of then given input
features = vgg.predict(x)
print decode_predictions(features, top=3)