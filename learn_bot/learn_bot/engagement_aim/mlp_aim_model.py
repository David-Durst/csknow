from typing import List
from torch import nn
import torch
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes


class MLPAimModel(nn.Module):
    internal_width = 1024
    cts: IOColumnTransformers
    output_layers: List[nn.Module]

    def __init__(self, cts: IOColumnTransformers):
        super(MLPAimModel, self).__init__()
        self.cts = cts
        self.inner_model = nn.Sequential(
            nn.Linear(cts.get_name_ranges(True, True)[-1].stop, self.internal_width),
            #nn.LeakyReLU(),
            #nn.Linear(self.internal_width, self.internal_width),
            #nn.LeakyReLU(),
            #nn.Linear(self.internal_width, self.internal_width),
            #nn.LeakyReLU(),
            nn.Linear(self.internal_width, cts.get_name_ranges(False, True)[-1].stop)
        )

    def forward(self, x):
        # transform inputs
        x_transformed = self.cts.transform_columns(True, x)

        # run model except last layer
        out_transformed = self.inner_model(x_transformed)

        # produce untransformed outputs
        out_untransformed = self.cts.untransform_columns(False, out_transformed)
        return torch.stack([out_transformed, out_untransformed], dim=1)

