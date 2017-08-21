
import keras
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dense, merge
from keras.models import Model
from keras.optimizers import Adam

'''
The purpose of this program to predict 4'th character using 3 previous characters with out using SimpleRNN API

Expanded architecture RNN:

	Out_put (softmax)
	  ^
	  |
 c3-> 0 (dense_hidden) (tanh)
 	  ^
	  | 
 c2-> 0 (dense_input) (relu)
 	  ^
 	  |
 	  c1

1) Convert all c1, c2, and c3 chars to the embeddings
2) Apply c1's, c2's embeddings to the dense layers (dense_input)
3) Merge c1 and c2 dense outputs (sum)=> c2_hidden
4) Apply c3's embedding to a dense layer (dense_hidden)
5) Merge the c2_hidden and c3's dense output
6) It generates a 4'th char (apply to the output Dense layer)
7) Let the model runs until loss function achieves the good result
'''


'''
Download the dataset from https://s3.amazonaws.com/text-datasets/nietzsche.txt
It contains english sentences
'''
file_text = open('<Base_Path>/nietzsche.txt', 'rb').read()

'''
Findig out the vocab size and using unique chararcters (labels)
'''
chars= sorted(list(set(file_text)))
vocab_size = len(chars) + 1


'''
Two dictionaries objects: First is to look-up for index to char, and Second is lookup for char to index
'''
index_chars = dict((i,c) for i, c in enumerate(chars))
chars_index = dict((c,i) for i, c in enumerate(chars))


'''Replace all chararcters with their indexes'''
idx = [chars_index[c] for c in file_text]
 

'''
Converting the dataset to the training dataset 3 chars (features inputs)
lst_char4: Label (output)
'''
lst_char1 = [idx[i] for i in xrange(0, len(idx)-1-3, 3)]
lst_char2 = [idx[i+1] for i in xrange(0, len(idx)-1-3, 3)]
lst_char3 = [idx[i+2] for i in xrange(0, len(idx)-1-3, 3)]
lst_char4 = [idx[i+3] for i in xrange(0, len(idx)-1-3, 3)]


'''
Conveting the lists to one dimensional arrays
'''
x1 = np.stack(lst_char1[:-2])
x2 = np.stack(lst_char2[:-2])
x3 = np.stack(lst_char3[:-2])
y = np.stack(lst_char4[:-2])


'''
This function is used for converting the input sequences to an Embedding
'''
def createEmbedding(name, n_in, n_out):
    inpt = Input(shape=(1,), dtype='int64', name=name)
    embd = Embedding(n_in, n_out, input_length=1)(inpt)
    flat = Flatten()(embd)
    return inpt, flat

c1_in, c1 = createEmbedding('c1', vocab_size, 42)
c2_in, c2 = createEmbedding('c2', vocab_size, 42)
c3_in, c3 = createEmbedding('c3', vocab_size, 42)

 

n_hidden = 256

dense_in = Dense(n_hidden, activation='relu')

dense_hidden = Dense(n_hidden, activation='tanh')

c1_hidden = dense_in(c1)



c3_hidden = dense_in(c3)


c2_dense = dense_in(c2)
hidden_2 = dense_hidden(c1_hidden)
c2_hidden = merge([c2_dense, hidden_2])

c3_dense =dense_in(c3)
hidden_3= dense_hidden(c2_hidden)
c3_hidden = merge([c3_dense, hidden_3])

out = Dense(vocab_size, activation='softmax')
c4_out = out(c3_hidden)



'''
Compiling and fiting the model
'''
model = Model([c1_in,c2_in,c3_in], c4_out)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
model.fit([x1,x2,x3], y, batch_size=64, epochs=10)


'''
Test case 1: Given three chars predict 4'th one by the model
'''

def test(chars):
    ids = [chars_index[c] for c in chars]
    arr = [np.array(i)[np.newaxis] for i in ids]
    pred = model.predict(arr)
    i = np.argmax(pred)
    return index_chars[i]


# In[85]:


test('joo')


# In[ ]:




