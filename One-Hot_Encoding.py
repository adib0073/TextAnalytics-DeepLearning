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