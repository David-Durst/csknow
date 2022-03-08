from torch import nn
import torch
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class NNArgs:
    player_id_to_ix: Dict[int, int]
    embedding_dim: int
    input_ct: ColumnTransformer
    output_ct: ColumnTransformer
    output_names: List[str]
    output_ranges: List[slice]


class NeuralNetwork(nn.Module):
    args: NNArgs

    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        self.args = args
        self.embeddings = nn.Embedding(len(args.player_id_to_ix), args.embedding_dim)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(args.embedding_dim + args.input_ct.get_feature_names_out().size - 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.moveLayer = nn.Sequential(
            nn.Linear(128, args.output_ranges[0].stop - args.output_ranges[0].start),
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
        embeds = self.embeddings(idx_long).view((-1, self.args.embedding_dim))
        x_all = torch.cat((embeds, x_vals), 1)
        logits = self.linear_relu_stack(x_all)
        moveOutput = self.moveLayer(logits)
        #moveOutput = self.moveSigmoid(self.moveLayer(logits))
        crouchOutput = self.crouchLayer(logits)
        shootOutput = self.shootLayer(logits)
        return torch.cat((moveOutput, crouchOutput, shootOutput), dim=1)
