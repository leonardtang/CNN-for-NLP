import gensim
import numpy as np
import torch


# glove_path = "kim_2014"
# Using GloVe embeddings with 50-dimensional features
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

self.embed = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
