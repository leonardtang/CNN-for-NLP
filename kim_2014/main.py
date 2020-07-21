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
embedding_dim = 300  # Default for Google's word2vec
input_channels = 1
filter_heights = [3, 4, 5]
filter_count = 100
dropout_p = 0.5
classes = 2

# Setting random seeds for data loading:
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Loading device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Make sure to pip install spacy and python -m spacy download en
# What even is spacy? Clarify
def load_imdb(TEXT=data.Field(tokenize='spacy'), LABEL=data.LabelField(dtype=torch.float)):
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, val_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))
    # TEXT.build_vocab(train)
    return train_data, val_data, test_data


def get_iterators(batch_size, train_data, val_data, test_data):
    train_iterator, val_iterator, test_iterator = torchtext.data.BucketIterator.splits(
                                                            (train_data, val_data, test_data),
                                                            batch_size=batch_size)
    return train_iterator, val_iterator, test_iterator


train_data, val_data, test_data = load_imdb()
print(train_data.examples[10], "\n--", val_data.examples[10], "\n --", test_data.examples[10])
print("Data loaded")
train_iterator, val_iterator, test_iterator = get_iterators(batch_size, train_data, val_data, test_data)
print("Iterators loaded")

net = model.KimModel(embedding_dim=embedding_dim, input_channels=input_channels,
                     filter_heights=filter_heights, filter_count=filter_count, dropout_p=dropout_p, classes=classes)
print("Model loaded")

train.train(net, device, train_iterator, val_iterator, batch_size, n_epochs, learning_rate)