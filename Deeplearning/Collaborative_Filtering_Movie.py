import pandas as pd
import numpy as np 
import keras
from keras.layers import Input,Embedding,merge,Flatten
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam

'''
Objective of this program to implement a Collaborative filtering for movie recommendation
The model architecture is:
First we take UserId and MovieId fields, represent them in a 50 dimensional vector spaces using a Embedding layer
After conversion, we perform the dot product between movieId and UserId (embedding layers) using a merge layer

We are using the MSE-- loss function and ADAM-- Optimizer
'''

#Download the movie from this location: https://grouplens.org/datasets/movielens/
csvreader = pd.read_csv('/Users/swapnilsoni/Downloads/ml-latest-small/ratings.csv')

# Print default 5 rows to see the data
print csvreader.head()

'''
 Get a number of unique usersID and moviesID from the training data set
 These values help to define the input dimensions of the Embedding layes
'''
nuser = csvreader.userId.nunique()
nmovieId = csvreader.movieId.nunique()
print nuser,nmovieId

print csvreader.userId.min(), csvreader.userId.max()
print csvreader.movieId.min(), csvreader.movieId.max()

#Preparing the training and validation set 80% and 20%
np.random.seed = 42
msk = np.random.rand(len(csvreader)) < 0.8
trn = csvreader[msk]
val = csvreader[~msk]


''' 
Here, I am using the Keras functional API --  Embedding, merge
UserEmbedding layer:
 Input layer: Only passing one dimension (i.e userID itseld)
 Embedding layer (UserID): 
    Input-dimension: A maximum value of the userID + 1 (because it starts from 0)
    Output dimension: I have decided to use 50 (you can decide accordingly)
    Input-length: It is just a userID, so the value is 1
 
Embedding layer (MovieId): 
    Input-dimension: A maximum value of the movieId + 1 (because it starts from 0)
    Output dimension: I have decided to use 50 (you can decide accordingly)
    Input-length: It is just a movieID, so the value is 1


'''
user_in = Input(shape=(1,), dtype='int64', name='user_in')
embedding_user = Embedding(input_dim=nuser+1, output_dim=50, input_length=1, embeddings_regularizer=l2(1e-5))(user_in)
movie_in = Input(shape=(1,), dtype='int64', name='movie_in')
embedding_movie = Embedding(input_dim=163949+1, output_dim=50, input_length=1, embeddings_regularizer=l2(1e-5))(movie_in)

#performing the dot product
x = merge([embedding_user, embedding_movie], mode='dot')
x = Flatten()(x)
model = Model([user_in, movie_in], x)
#Compiling and learning
model.compile(Adam(0.001), loss="mse")
model.fit([trn.userId, trn.movieId], trn.rating, batch_size=64,epochs=6,validation_data=([val.userId, val.movieId], val.rating))