%matplotlib inline
import keras
from keras.datasets import mnist
import matplotlib
from matplotlib import pyplot
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#download the minst data sets
(x_data, y_data), (x_test, y_test) = mnist.load_data()

# checking keras image format "channels_last" or "channels_first
print K.image_data_format()
#reshaping the image data set from 2D to 3D 
x_data = x_data.reshape(x_data.shape[0], 1, x_data.shape[1], x_data.shape[2])
x_test= x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
# printing 9 images
for i in range(0, 9):
    pyplot.subplot(440 + 1 + i)
    pyplot.imshow(x_data[i])

#Normalizing and converting image array
x_data.astype('float32')
x_data = x_data / 255.0
x_test.astype('float32')
x_test = x_test /255.0


# converting labels to one hot vector
y_data = np_utils.to_categorical(y_data)
y_test = np_utils.to_categorical(y_test)


# very simple convolution network: one conv2D+maxpoling+dense(with 10 output)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
# accuracy : metrics to evaluate the model
# crossentropy: loss function
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_data,y_data,epochs=10,batch_size=128,validation_split=0.3)


model.evaluate(x_test,y_test)