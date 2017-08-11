'''
This program uses word embedding to predict sentiment of movie reviews.
Architecture:
1) Embedding layer 
2) CNN 
3) Dense
'''
import keras
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, Dropout
from keras.optimizers import Adam

'''
Download the IMBD training and testing data set
'''
(x_train, y_train), (x_test,y_test) = imdb.load_data()

'''
This will return the word to index mapping
'''
idx = imdb.get_word_index()

'''
Here, I am converting index to word mapping
'''
ind2Wrd = {i:w for w, i in idx.iteritems()}


'''
As there so many words are available in the IMDB data set, so we are only limitin our self to 5000 indexes
Setting index 5000 for all thoses words which index > 5000 
'''
vocab_size = 5000
x_train = [np.array([val if val < vocab_size - 1 else vocab_size -1 for val in row]) for row in x_train]
x_test = [np.array([val if val < vocab_size- 1 else vocab_size -1 for val in row]) for row in x_test]


'''
Converting list to 2D numpy array. 
Maxlen attribute allows us to limit the number of words to 500 in a review. It pads with zero if the length is less than 500
'''
x_train= sequence.pad_sequences(x_train, maxlen=500, value=0)
x_test = sequence.pad_sequences(x_test, maxlen=500, value=0)


input_dim = max(sorted(idx.values()))

model = Sequential([
 Embedding(5000, 32, input_length = 500, dropout=0.2),
 Dropout(0.2),
 Conv1D(64, 5, padding="same", activation='relu'),
 MaxPooling1D(),
 Flatten(),
 Dense(100,activation ='relu'),
 Dropout(0.7),
 Dense(1, activation='sigmoid') 
])



model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()


model.fit(x_train, y_train, validation_data=(x_test,y_test), nb_epoch=2, batch_size=64)




