import torch
import torch.nn.functional as F
import gensim


class KimModel(torch.nn.Module):
    """ One layer CNN with (def=100) filters of (def=[3,4,5]) varying heights; dropout and L2 weight constraint """

    # Takes in word2vec embeddings from main
    def __init__(self, embedding_dim=300, input_channels=1, filter_heights=[3, 4, 5],
                 filter_count=100, dropout_p=0.5, classes=2):

        super(KimModel, self).__init__()

        # Embeddings: the input to the module is a list of indices, and the output is the corresponding feature vectors
        # Using word2vec features pre-trained on Google News corpus
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                binary=True,
                                                                unicode_errors='ignore')
        weights = torch.FloatTensor(model.vectors)
        self.embed = torch.nn.Embedding.from_pretrained(torch.FloatTensor(weights))
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

