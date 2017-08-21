import keras
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dense, merge
from keras.models import Model
from keras.optimizers import Adam

'''
The purpose of this program to predict 9'th character using 8 previous characters with out using SimpleRNN API in Keras

Closed architecture RNN:

    Out_put (softmax)
       ^
       |
       | Loop (2 --> n-1)
|-------------------------------|
| cn-> 0 -------|               |
|      ^        |               |
|      |        |               |
|      | <------'               |
|      |                        |
 --------------------------------
       0 (dense_input) (relu)
       ^
       |
       c1

1) Convert all c1 to c8 chars to the embeddings
2) Apply c1's to a dense layer (dense_input)
3) Start loop from c2 to c8
 3.1)  Merge c1 and c2 dense outputs (sum)=> hidden
 3.2)  Apply c3 embedding to the dense layes, and merge with previous step (3.1) hidden output
 3.3) Continues this process until all emdedding are finished

6) Finally the output should be applied to the output dense layer
7) Let the model runs until loss function achieves the good result
'''


'''
Download the dataset from https://s3.amazonaws.com/text-datasets/nietzsche.txt
It contains english sentences
'''
file_text = open('<Base_Path>/nietzsche.txt','rb').read()

'''
Findig out the vocab size and using unique chararcters (labels)
'''
chars=sorted(list(set(file_text)))
vocab_size = len(chars) + 1

'''
Two dictionaries objects: First is to look-up for index to char, and Second is lookup for char to index
'''
index_chars = dict((i,c) for i, c in enumerate(chars))
chars_index = dict((c,i) for i, c in enumerate(chars))

'''Replacing all the chars with their indexes'''
idx = [chars_index[c] for c in file_text]

'''
Converting the dataset to the training dataset
xs : It is a sequence of 7 chars (features inputs)
c_cout: Every 8'th char is Label (output)
'''
c_in = [[idx[i+n] for i in xrange(0, len(file_text)-1-8, 8)] for n in range(0, 8)]
c_cout = [[idx[i+n] for i in xrange(1, len(file_text)-1-8, 8)] for n in range(1, 8)]

xs = [np.stack(c[:-2]) for c in c_in]
y= np.stack(c_cout[:-2])


print len(c_in)
print xs[0].shape


def createEmbedding(name, n_in, n_out):
    inpt = Input(shape=(1,), dtype='int64', name=name)
    embd = Embedding(n_in, n_out, input_length=1)(inpt)
    flat = Flatten()(embd)
    return inpt, flat

c_ins = [createEmbedding('c_'+str(n), vocab_size, 42) for n in range(8)]



dense_in = Dense(256, activation='relu')
dense_hidden = Dense(256, activation='relu', kernel_initializer='identity')
dense_out = Dense(vocab_size, activation='softmax')


hidden = dense_in(c_ins[0][1])


for i in range(1, 8):
    c_dense = dense_in(c_ins[i][1])
    hidden = dense_hidden(hidden)
    hidden = merge([hidden, c_dense])
out = dense_out(hidden)

'''
Compiling and fiting the model
'''
model = Model([c[0] for c in c_ins],out)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
model.fit(xs, y, batch_size=64, epochs=10)


'''
Test case 1: Given three chars predict 4'th one by the model
'''
def test(chars):
    ids = [chars_index[c] for c in chars]
    arr = [np.array(i)[np.newaxis] for i in ids]
    pred = model.predict(arr)
    i = np.argmax(pred)
    return index_chars[i]


test('I love y')




