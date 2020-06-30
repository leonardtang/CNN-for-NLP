import torch
import torch.nn.functional as F


class KimModel(torch.nn.Module):
    """ One layer CNN with (def=100) filters of (def=[3,4,5]) varying heights; dropout and L2 weight constraint """

    # Takes in word2vec embeddings from main
    def __init__(self):  # Probably should have default param values here
        super(KimModel, self).__init__()

        num_embeddings = 500  # THIS SHOULD COME FROM MAIN (length of input words)
        embedding_dim = 300  # Default for Google's word2vec
        input_channels = 1  # Input is a single Word Count x Feature Dimension matrix
        filter_heights = [3, 4, 5]  # Suggested heights from the paper
        filter_count = 100  # Default from paper (100 feature maps per filter height)
        dropout_p = 0.5  # Paper default
        classes = 2

        # Careful about embeddings (basically dictionary mapping words to vectors)
        # Embeddings: the input to the module is a list of indices, and the output is the corresponding word embeddings
        self.embed = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # There are (filter_heights) sets of (filter_count) different channels IN PARALLEL
        self.convs1 = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=input_channels,
                                                           out_channels=filter_count,
                                                           kernel_size=(H, embedding_dim)) for H in filter_heights])

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc = torch.nn.Linear(len(filter_heights) * filter_count, classes)

    def forward(self, x):
        x = self.embed(x)  # Input :==> (N=batch, H=embeddings, W=embed_dim)
        x = torch.unsqueeze(x, 1)  # Output dims: (N, 1, H, W)
        # Note convolution flattens matrix into vector
        x = [conv(x) for conv in self.convs1]  # Output dims: [(N, filter_count, H, 1), ...]*len(filter_heights)
        x = [F.relu(x_i) for x_i in x]
        x = [x_i.squeeze(3) for x_i in x]  # Output dims: [(N, filter_count, H), ...]*len(filter_heights)
        # x_i.size(2) is height of feature vector -- so we're performing max-over-time pooling (window is entire vector)
        x = [F.max_pool1d(x_i, x_i.size(2)) for x_i in x]  # Output dims: [(N, filter_count, 1),...]*len(filter_heights)
        x = [x_i.squeeze(2) for x_i in x]  # Output dims: [(N, filter_count),...]*len(filter_heights)
        x = torch.cat(x, 1)  # Output dims: (N, filter_count x len(filter_heights) == output_channels)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

