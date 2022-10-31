from typing import List
from torch import nn
import torch
from learn_bot.navigation.io_transforms import IOColumnAndImageTransformers, ColumnTypes


class CNNNavModel(nn.Module):
    internal_width = 1024
    cts: IOColumnAndImageTransformers
    output_layers: List[nn.Module]

    def __init__(self, cts: IOColumnAndImageTransformers):
        super().__init__()
        self.cts = cts
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        # need to call this once I have the CT, use those dims to implement conv dims
        raise NotImplementedError
        self.inner_model = nn.Sequential(
            #nn.Linear(cts.get_name_ranges(True, True)[-1].stop, self.internal_width),
            #nn.ReLU(),
            #nn.Linear(self.internal_width, self.internal_width),
            #nn.ReLU(),
            #nn.Linear(self.internal_width, self.internal_width),
            #nn.ReLU(),
        )

        output_layers = []
        for output_range in cts.get_name_ranges(False, True):
            output_layers.append(nn.Linear(self.internal_width, len(output_range)))
        self.output_layers = nn.ModuleList(output_layers)

    # https://github.com/pytorch/pytorch/issues/18337
    # accepting multiple inputs is fine in C++ side too
    def forward(self, non_img_x, img_x):
        # transform inputs
        x_transformed = self.cts.transform_images_and_columns(True,non_img_x, img_x)
        raise NotImplementedError

        # run model except last layer
        logits = self.inner_model(x_transformed)

        # produce transformed outputs
        outputs = []
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

