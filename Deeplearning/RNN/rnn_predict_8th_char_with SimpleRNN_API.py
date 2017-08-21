import keras
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dense, merge,TimeDistributed
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam

file_text = open('<Base_Path>/nietzsche.txt', 'rb').read()


chars= sorted(list(set(file_text)))
vocab_size = len(chars) + 1



index_chars = dict((i,c) for i, c in enumerate(chars))
chars_index = dict((c,i) for i, c in enumerate(chars))
idx = [chars_index[c] for c in file_text]
c_in = [[idx[i+n] for i in xrange(0, len(file_text)-1-8, 8)] for n in range(8)]
c_cout = [[idx[i+n] for i in xrange(1, len(idx)-8, 8)] for n in range(8)]


n_hidden=256
model = Sequential([
    Embedding(vocab_size, 42, input_length=8),
    SimpleRNN(n_hidden,return_sequences=True, activation='relu', inner_init='identity'),
    TimeDistributed(Dense(vocab_size, activation='softmax'))
])


model.summary()


xs = [np.stack(c[:-2]) for c in c_in]
ys = [np.stack(c[:-2]) for c in c_cout]
xs_rnn =np.stack(np.squeeze(xs), axis=1)
ys_rnn = np.stack(ys, axis=1)[np.newaxis]
ys_rnn = np.stack(ys_rnn, axis=2)


xs_rnn.shape, ys_rnn.shape


model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())


model.fit(xs_rnn, ys_rnn, batch_size=32, epochs=10)

def get_next_char(chars):
    ids = [chars_index[c] for c in chars]
    arr = np.array(ids)[np.newaxis,:]
    pred = model.predict(arr)[0]
    print(list(chars))
    return [index_chars[np.argmax(o)] for o in pred]


# In[107]:


get_next_char(' is this')


# In[ ]:




