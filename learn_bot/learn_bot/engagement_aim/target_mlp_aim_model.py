from typing import List
from torch import nn
from learn_bot.libs.io_transforms import IOColumnTransformers


class TargetMLPAimModel(nn.Module):
    internal_width = 1024
    target_width = 2
    cts: IOColumnTransformers
    output_layers: List[nn.Module]

    def __init__(self, cts: IOColumnTransformers):
        super(TargetMLPAimModel, self).__init__()
        self.cts = cts
        self.target_model = nn.Sequential(
            nn.Linear(cts.get_name_ranges(True, True)[-1].stop, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.target_width)
        )
        self.movement_model = nn.Sequential(
            nn.Linear(self.target_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, cts.get_name_ranges(False, True)[-1].stop)
        )

    def forward(self, x):
        # transform inputs
        x_transformed = self.cts.transform_columns(True, x, x)

        # run model except last layer
        target_transformed = self.target_model(x_transformed)
        out_transformed = self.movement_model(target_transformed)
        #out_transformed = torch.cat([target_transformed, movement_transformed], dim=1)

        # produce untransformed outputs
        out_untransformed = self.cts.untransform_columns(False, out_transformed, x)
        # https://github.com/pytorch/pytorch/issues/22440 how to parse tuple output
        return out_transformed, out_untransformed

