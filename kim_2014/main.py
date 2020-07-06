import torch
import random
import torchtext
import model
import train
from torchtext import data
from torchtext import datasets

# Training hyper-parameters
batch_size = 16
n_epochs = 20
learning_rate = 1.0

# Architecture hyper-parameters
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
random.setstate(SEED)
torch.backends.cudnn.deterministic = True

# Loading device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_imdb(TEXT=data.Field(tokenize='spacy'), LABEL=data.LabelField(dtype=torch.float)):
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, val_data = train_data.split(split_ratio=0.7, random_state=random.seed(SEED))
    return train_data, val_data, test_data


def get_iterators(batch_size, train_data, val_data, test_data):
    train_iterator, val_iterator, test_iterator = torchtext.data.BucketIterator.splits(
                                                            (train_data, val_data, test_data),
                                                            batch_size=batch_size)
    return train_iterator, val_iterator, test_iterator


""" Figure out word2vec tokinization? """

net = model.KimModel()

