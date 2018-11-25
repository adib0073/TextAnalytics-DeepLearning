from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

'''
# Simple RNN

model = Sequential()
model.add(Embedding(10000, 32)) 
model.add(SimpleRNN(32)) ### With Full State sequence it will be ### model.add(SimpleRNN(32, return_sequences=True))
model.summary()

'''
#Considering stack of Simple RNNs
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()
