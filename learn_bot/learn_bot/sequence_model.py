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
    internal_width = 64
    args: SNNArgs

    def __init__(self, args: SNNArgs):
        super(SequenceNeuralNetwork, self).__init__()
        self.args = args
        self.embeddings = nn.Embedding(len(args.player_id_to_ix), args.embedding_dim)

        self.inner_model = nn.LSTM(args.embedding_dim + args.input_ct.get_feature_names_out().size - 1,
                                   self.internal_width,
                                   3, batch_first=True, dropout=0.2)

        output_layers = []
        for output_range in args.output_ranges:
            output_layers.append(nn.Linear(self.internal_width, len(output_range)))
        self.output_layers = nn.ModuleList(output_layers)
        #self.moveSigmoid = nn.Sigmoid()

    def forward(self, x, lens):
        inner_most_dimension_idx = len(x.shape) - 1
        idx, x_vals = x.split([1, x.shape[inner_most_dimension_idx] - 1], dim=inner_most_dimension_idx)
        idx_long = idx.long()
        embeds = self.embeddings(idx_long).view((idx.shape[0], idx.shape[1], self.args.embedding_dim))
        embeds_zero = torch.zeros_like(embeds)
        x_all = torch.cat((embeds_zero, x_vals), 2)

        x_packed = pack_padded_sequence(x_all, lens, batch_first=True, enforce_sorted=False)
        # ignore inner state since doing complete sequence each time
        logits_packed, _ = self.inner_model(x_packed)
        # ignoring output seqs since already have those from lens
        logits, _ = pad_packed_sequence(logits_packed, batch_first=True)

        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(logits))
        return torch.cat(outputs, dim=2)
