from typing import List
from torch import nn
import torch
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes, FUTURE_TICKS, CUR_TICK


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, X):
        # src = [batch size, src len, features per batch]
        outputs, (hidden, cell) = self.lstm(X)

        # outputs = [batch size, src len, hid dim]
        # hidden = [batch size, n layers, hid dim]
        # cell = [batch size, n layers, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size, 1]
        # unsqueeze input to add inner dimension of 1
        # sequence length will always be 1 as need to produce one output
        # as input to next one

        input = input.unsqueeze(1)

        output, (hidden, cell) = self.lstm(input, (hidden, cell))

        # outputs = [batch size, src len, hid dim]
        # hidden = [batch size, n layers, hid dim]
        # cell = [batch size, n layers, hid dim]

        # prediction = [batch size, output dim]
        return output.squeeze(1), hidden, cell

class LSTMAimModel(nn.Module):
    internal_width = 1024
    cts: IOColumnTransformers
    output_layers: List[nn.Module]

    def __init__(self, cts: IOColumnTransformers):
        super(LSTMAimModel, self).__init__()
        self.cts = cts
        self.encoder = Encoder(cts.get_name_ranges(True, True)[-1].stop, self.internal_width, 2, 0.2)
        self.decoder = Decoder(self.internal_width, 2, 0.2)

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        output_layers = []
        for output_range in cts.get_name_ranges(False, True):
            output_layers.append(nn.Linear(self.internal_width, len(output_range)))
        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, x):
        # transform inputs
        x_transformed = self.cts.transform_columns(True, x)

        # tensor to store decoder outputs
        outputs = []

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x_transformed)

        # first input to the decoder is the <sos> tokens
        input = x_transformed[:, -1]

        for t in range(0, CUR_TICK + FUTURE_TICKS):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs.append = output

        # produce transformed outputs
        for output_layer in self.output_layers:
            outputs.append(output_layer(logits))

        # produce untransformed outputs
        out_transformed = torch.cat(outputs, dim=1)
        out_untransformed = self.cts.untransform_columns(False, out_transformed)
        return torch.cat([out_transformed, out_untransformed], dim=1)

    def get_transformed_outputs(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :len(self.cts.output_types.column_names())]

    def get_untransformed_outputs(self, x: torch.Tensor):
        return x[:, len(self.cts.output_types.column_names()):]

