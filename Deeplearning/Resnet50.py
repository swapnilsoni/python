'''
 Purpose of this program to demostrate the use of ResNet. For this, I am using pretrained model of keras called ResNet50
'''
import keras
from keras.applications import resnet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import h5py as h5py

'''
Download the ResNet50 model without a top layer
'''
rsnet = resnet50.ResNet50(include_top=False)

'''
Make sure all the layers of the model should not be trained
'''
for layer in rsnet.layers:
    layer.tainable = False


'''
Attaching a new top layer(GlobalAveragePooling2D) to the model
'''
def get_Model():
    x = rsnet.output
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(2, activation='softmax')(x)
    return Model(inputs= rsnet.input, outputs=prediction)

model = get_Model()


'''
Preparing training and validation data set using ImageDataGenerator
'''
path = '<PATH>/data/'
datagen_train = ImageDataGenerator()
datagen_test = ImageDataGenerator()
generator_train = datagen.flow_from_directory(path +'train')
generator_test = datagen.flow_from_directory(path +'valid')


'''
training the model
'''

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])
model.fit_generator(generator, epochs=3, steps_per_epoch=10 ,validation_data=generator_test, validation_steps=10)
