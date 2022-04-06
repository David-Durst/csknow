from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class SNNArgs:
    player_id_to_ix: Dict[int, int]
    embedding_dim: int
    input_ct: ColumnTransformer
    output_ct: ColumnTransformer
    output_names: List[str]
    output_ranges: List[range]

class SequenceNeuralNetwork(nn.Module):
    internal_width = 1024
    args: SNNArgs

    def __init__(self, args: SNNArgs):
        super(SequenceNeuralNetwork, self).__init__()
        self.args = args
        self.embeddings = nn.Embedding(len(args.player_id_to_ix), args.embedding_dim)

        self.inner_model = nn.LSTM(args.embedding_dim + args.input_ct.get_feature_names_out().size - 1,
                                   self.internal_width,
                                   3, batch_first=True, dropout=0.5)

        output_layers = []
        for output_range in args.output_ranges:
            output_layers.append(nn.Linear(self.internal_width, len(output_range)))
        self.output_layers = nn.ModuleList(output_layers)
        #self.moveSigmoid = nn.Sigmoid()
        self.hn = None
        self.cn = None

    def reset_state(self):
        self.hn = None
        self.cn = None

    def forward(self, x, lens):
        idx, x_vals = x.split([1, x.shape[1] - 1], dim=1)
        idx_long = idx.long()
        embeds = self.embeddings(idx_long).view((-1, self.args.embedding_dim))
        x_all = torch.cat((embeds, x_vals), 1)

        x_packed = pack_padded_sequence(x_all, lens, batch_first=True, enforce_sorted=False)
        if self.hn is not None:
            logits_packed, (self.hn, self.cn) = self.inner_model(x_packed, (self.hn, self.cn))
        else:
            logits_packed, (self.hn, self.cn) = self.inner_model(x_packed)
        logits = pad_packed_sequence(logits_packed, batch_first=True)

        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(logits))
        return torch.cat(outputs, dim=1)
