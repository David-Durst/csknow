from typing import List
from torch import nn
from learn_bot.libs.io_transforms import IOColumnTransformers
import torch


class LSTMLatentModel(nn.Module):
    internal_width = 128
    cts: IOColumnTransformers
    output_layers: List[nn.Module]

    def __init__(self, cts: IOColumnTransformers):
        super(LSTMLatentModel, self).__init__()
        self.cts = cts
        self.inner_model = nn.Sequential(
            nn.LSTM(cts.get_name_ranges(True, True)[-1].stop, self.internal_width,
                    2, batch_first=True, dropout=0.2),
            nn.Linear(self.internal_width, cts.get_name_ranges(False, True)[-1].stop)
        )

    def forward(self, x):
        # transform inputs
        flattened_x = torch.flatten(x, 0, 1)
        x_transformed = self.cts.nested_transform_columns(True, flattened_x, flattened_x)

        # run model except last layer
        out_transformed = self.inner_model(x_transformed)

        # produce untransformed outputs
        out_untransformed = self.cts.nested_untransform_columns(False, out_transformed, flattened_x)
        # https://github.com/pytorch/pytorch/issues/22440 how to parse tuple output
        return out_transformed, out_untransformed

