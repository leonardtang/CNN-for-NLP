import torch
import random
import torchtext
import matplotlib.pyplot as plt
import numpy as np
import model
import train
import test
from torchtext import data
from torchtext import datasets

# Training hyper-parameters
batch_size = 2
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)


# Make sure to pip install spacy and python -m spacy download en
# What even is spacy? Clarify
# MUST have batch_first=TRUE to have dimensions be of form (batch, height) (for inputting into model)
def load_imdb(TEXT=data.Field(tokenize='spacy', batch_first=True),
              LABEL=data.LabelField(dtype=torch.long, batch_first=True)):
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, val_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))
    TEXT.build_vocab(train_data, vectors="glove.6B.300d", max_size=1000, min_freq=10)
    LABEL.build_vocab(test_data)
    embedding_layer = torch.nn.Embedding.from_pretrained(TEXT.vocab.vectors)
    return train_data, val_data, test_data, embedding_layer


def get_iterators(batch_size, train_data, val_data, test_data):
    train_iterator, val_iterator, test_iterator = torchtext.data.BucketIterator.splits(
                                                            (train_data, val_data, test_data),
                                                            batch_size=batch_size)
    return train_iterator, val_iterator, test_iterator


train_data, val_data, test_data, embedding_layer = load_imdb()
print("Data loaded")
train_iterator, val_iterator, test_iterator = get_iterators(batch_size, train_data, val_data, test_data)
print("Iterators loaded")

net = model.KimModel(embedding_layer=embedding_layer, embedding_dim=embedding_dim, input_channels=input_channels,
                     filter_heights=filter_heights, filter_count=filter_count, dropout_p=dropout_p, classes=classes)
print("Model loaded")

print("GPU count:", torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net, device_ids=[1, 2, 3, 0])
net = net.to(device)

print("Model sent")

model_state_dict, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
    train.train(net, device, train_iterator, val_iterator, batch_size, n_epochs, learning_rate)
torch.save(model_state_dict(), "model_weights.pth")

# Plotting loss and accuracy history
fig, (ax1, ax2) = plt.subplots(2)

# Loss history
ax1.set_title("Loss vs. Number of Training Epochs")
ax1.set(xlabel="Training Epoch", ylabel="Loss")
ax1.plot(range(1, len(train_loss_hist) + 1), train_loss_hist, label="Training")
ax1.plot(range(1, len(val_loss_hist) + 1), val_loss_hist, label="Validation")
print(np.concatenate((train_loss_hist, val_loss_hist)))
print(np.amax(np.concatenate((train_loss_hist, val_loss_hist))))
ax1.set_ylim((0, 1.25 * np.amax(np.concatenate((train_loss_hist, val_loss_hist), axis=0, out=None)).detach().cpu()))
ax1.set_xticks(np.arange(1, n_epochs + 1, 1.0))
ax1.legend()

# Accuracy history
ax2.set_title("Accuracy vs. Number of Training Epochs")
ax2.set(xlabel="Training Epoch", ylabel="Accuracy")
ax2.plot(range(1, n_epochs + 1), train_acc_hist, label="Training")
ax2.plot(range(1, n_epochs + 1), val_acc_hist, label="Validation")
ax2.set_ylim(0, 100)
ax2.set_xticks(np.arange(1, n_epochs + 1, 1.0))
ax2.legend()

plt.tight_layout()
plt.savefig("CNN-for-NLP-Trial-1.png")

test_acc = test.test(net, device, test_iterator)
print(test_acc)