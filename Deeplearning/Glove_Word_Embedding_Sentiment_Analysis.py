'''
This program uses existing word embedding, Glove, to predict sentiment of movie reviews.
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
Download the glove word embedding
'''
glove_base_dir = '<PATH>/glove.6B/'
file_open = open(glove_base_dir + 'glove.6B.50d.txt')

words_index = {}
vectors = []
index = 0
'''
Create word to index, and separate out the vector from the glove embedding
'''
for line in file_open:
    values = line.split()
    word = values[0]
    words_index[word] = index
    vectors.append(np.asarray(values[1:], dtype ='float32'))
    index = index + 1

vectors = np.asarray(vectors)


def word2vec(word): return vectors[words_index[word]]



'''
As there so many words are available in the IMDB data set, so we are only limitin our self to 5000 indexes
Setting index 5000 for all thoses words which index > 5000 
'''
vocab_size = 5000
x_train = [np.array([val if val < vocab_size - 1 else vocab_size -1 for val in row]) for row in x_train]
x_test = [np.array([val if val < vocab_size- 1 else vocab_size -1 for val in row]) for row in x_test]
x_train= sequence.pad_sequences(x_train, maxlen=500, value=0)
x_test = sequence.pad_sequences(x_test, maxlen=500, value=0)


'''
This function is used for retrieving vectors from glve embedding of the imdb words
'''
rows_count = vectors.shape[1]
def create_embd():
    emd = np.zeros((vocab_size, rows_count))
    for i in range(1, len(emd)):
         if words_index.get(ind2Wrd[i]):
            emd[i] = vectors[words_index.get(ind2Wrd[i])]
         else:
            emd[i] = np.random.normal(scale=0.6, size=(rows_count,))
    return emd

embdd = create_embd() 


model = Sequential([
 Embedding(5000, 50, input_length = 500, dropout=0.2, weights=[embdd], trainable=False),
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