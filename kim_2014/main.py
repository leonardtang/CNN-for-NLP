import torch.nn as nn
import torch
import random
from torchtext import data
from torchtext import datasets

# Hyper-parameters
batch_size = 16
n_epochs = 20
learning_rate = 1.0

# Architecture parameters
num_embeddings = 500  # Length of input phrase
embedding_dim = 300  # Default for Google's word2vec
input_channels = 1
filter_heights = [3, 4, 5]
filter_count = 100
dropout_p = 0.5
classes = 2

# Setting random seeds for data loading:
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def load_imdb(TEXT=data.Field(tokenize='spacy'), LABEL=data.LabelField(dtype=torch.float)):
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, val_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))
    return train_data, test_data


def get_train_loader(batch_size, train_set, train_sampler):
    """ Gets a batch sample of training data; called during training process for each batch """
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    return train_loader


def get_val_loader(train_set, test_set, val_sampler):
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=128, sampler=val_sampler, num_workers=8)
    return val_loader



""" Figure out word2vec tokinization? """

""" BucketIterator """

model