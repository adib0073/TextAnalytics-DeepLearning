# Word - level one hot encoding

import numpy as np
samples = ['Hello World', 'How are you.']
token_index = {}
for sample in samples:
    print(sample)
    for word in sample.split():
        print(word)
        if word not in token_index:
            token_index[word] = len(token_index) + 1
max_length = 10
results = np.zeros(shape=(len(samples),max_length,max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1
        
print(token_index) # {'World': 2, 'Hello': 1, 'you.': 5, 'are': 4, 'How': 3}
results.shape #(2, 10, 6)  ## ( num_sentences,max_length,total_token_indices)

# Character-level one-hot encoding

import string
samples = ['Hello World', 'How are you.']
characters = string.printable   #All printable ASCII characters
token_index = dict(zip(range(1, len(characters) + 1), characters))
max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1
        
        
 
