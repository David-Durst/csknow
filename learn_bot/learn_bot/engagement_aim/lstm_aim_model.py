from typing import List
from torch import nn
import torch
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes, PRIOR_TICKS, FUTURE_TICKS, \
    CUR_TICK, ColumnTransformerType


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

        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(hid_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size, seq len = 1, hid dim]
        # sequence length will always be 1 as need to produce one output
        # as input to next one
        output, (hidden, cell) = self.lstm(input, (hidden, cell))

        # outputs = [batch size, seq len = 1, hid dim]
        # hidden = [batch size, n layers, hid dim]
        # cell = [batch size, n layers, hid dim]
        return output, hidden, cell

class LSTMAimModel(nn.Module):
    internal_width = 256
    cts: IOColumnTransformers
    output_layers: List[nn.Module]
    num_categorical_transformed_features: int
    num_prior_ticks: int
    num_input_temporal_features: int
    input_temporal_data_per_tick: int

    def __init__(self, cts: IOColumnTransformers):
        super(LSTMAimModel, self).__init__()
        self.cts = cts
        self.num_categorical_transformed_features = \
            len(self.cts.get_name_ranges(True, True, {ColumnTransformerType.CATEGORICAL})[-1])
        self.num_prior_ticks = -1 * PRIOR_TICKS
        self.num_input_temporal_features = len(self.cts.input_types.float_standard_cols)
        self.input_temporal_data_per_tick = int(self.num_input_temporal_features / self.num_prior_ticks)

        self.encoder = Encoder(self.input_temporal_data_per_tick + self.num_categorical_transformed_features,
                               self.internal_width, 1, 0.2)
        self.decoder = Decoder(self.internal_width, 1, 0.2)

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

        # convert data from [batch len, data per tick * num prior ticks + constant data]
        # to [batch _len, prior tics, data per tick + constant data]
        # this will hold the temporal data and the static data duplicated for each time step
        #x_transformed_all_data = torch.zeros([x.shape[0], self.num_prior_ticks,
        #                                      input_temporal_data_per_batch + self.num_categorical_transformed_features])
        x_transformed_temporal = x_transformed[:, :self.num_input_temporal_features]
        x_transformed_temporal_per_tick = x_transformed_temporal.reshape([-1, self.num_prior_ticks,
                                                                          self.input_temporal_data_per_tick])
        #x_transformed_all_data[:, :, :input_temporal_data_per_batch] = x_transformed_temporal_per_tick
        x_transformed_non_temporal = x_transformed[:, self.num_input_temporal_features:]
        x_transformed_non_temporal_duplicated = \
            x_transformed_non_temporal.reshape(-1, 1, self.num_categorical_transformed_features) \
            .expand([-1, self.num_prior_ticks, -1])
        x_transformed_all_data = torch.cat([x_transformed_temporal_per_tick, x_transformed_non_temporal_duplicated], dim=2)

        # tensor to store decoder outputs
        outputs = []

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_output, hidden, cell = self.encoder(x_transformed_all_data)

        # first input to the decoder is last known value
        decoder_input = encoder_output[:, -1, :].unsqueeze(1)#x_transformed_all_data[:, -1, :self.num_input_temporal_features]

        # produce transformed outputs
        for t in range(0, CUR_TICK + FUTURE_TICKS):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # place predictions in a tensor holding predictions for each time step
            for c in range(self.input_temporal_data_per_tick):
                outputs.append(self.output_layers[t*self.input_temporal_data_per_tick + c](decoder_output))

            # feed outputs back as inputs
            decoder_input = decoder_output

        # no constant output predictions, so no conversion necessary, cat will just combine all outputs

        # produce untransformed outputs
        out_transformed = torch.cat(outputs, dim=1).squeeze(dim=2)
        out_untransformed = self.cts.untransform_columns(False, out_transformed)
        return torch.cat([out_transformed, out_untransformed], dim=1)

    def get_transformed_outputs(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :len(self.cts.output_types.column_names())]

    def get_untransformed_outputs(self, x: torch.Tensor):
        return x[:, len(self.cts.output_types.column_names()):]

