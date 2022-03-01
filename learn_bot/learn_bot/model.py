from torch import nn
import torch

class NeuralNetwork(nn.Module):
    embedding_dim: int

    def __init__(self, num_unique_players, embedding_dim, input_ct, output_ranges):
        super(NeuralNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_unique_players, embedding_dim)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embedding_dim + input_ct.get_feature_names_out().size - 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.moveLayer = nn.Sequential(
            nn.Linear(128, output_ranges[0].stop - output_ranges[0].start),
        )
        self.crouchLayer = nn.Sequential(
            nn.Linear(128, 2),
        )
        self.shootLayer = nn.Sequential(
            nn.Linear(128, 2),
        )
        #self.moveSigmoid = nn.Sigmoid()

    def forward(self, x):
        idx, x_vals = x.split([1, x.shape[1] - 1], dim=1)
        idx_long = idx.long()
        embeds = self.embeddings(idx_long).view((-1, self.embedding_dim))
        x_all = torch.cat((embeds, x_vals), 1)
        logits = self.linear_relu_stack(x_all)
        moveOutput = self.moveLayer(logits)
        #moveOutput = self.moveSigmoid(self.moveLayer(logits))
        crouchOutput = self.crouchLayer(logits)
        shootOutput = self.shootLayer(logits)
        return torch.cat((moveOutput, crouchOutput, shootOutput), dim=1)
