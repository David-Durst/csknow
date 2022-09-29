from typing import List
from torch import nn
import torch
from learn_bot.engagement_aim.column_management import IOColumnTransformers


class LinearModel(nn.Module):
    internal_width = 1024
    cts: IOColumnTransformers
    output_layers: List[nn.Module]

    def __init__(self, cts: IOColumnTransformers):
        super(LinearModel, self).__init__()
        self.cts = cts
        self.inner_model = nn.Sequential(
            nn.Linear(cts.input_ct.get_feature_names_out().size, self.internal_width)
        )

        output_layers = []
        for output_range in cts.get_name_ranges(False):
            output_layers.append(nn.Linear(self.internal_width, len(output_range)))
        self.output_layers = nn.ModuleList(output_layers)

    def convert_output_labels_for_loss(self, y):
        ys_transformed = []
        for i, output_range in enumerate(self.cts.get_name_ranges(False)):
            ys_transformed.append(self.cts.output_ct_pts[i].convert(y[:, output_range]))
        return torch.cat(ys_transformed, dim=1)

    def forward(self, x):
        # transform inputs
        xs_transformed = []
        for i, input_range in enumerate(self.cts.get_name_ranges(True)):
            xs_transformed.append(self.cts.input_ct_pts[i].convert(x[:, input_range]))
        x_transformed = torch.cat(xs_transformed, dim=1)

        # run model except last layer
        logits = self.inner_model(x_transformed)

        # produce transformed outputs
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(logits))

        # produce untransformed outputs
        for i in range(len(self.output_layers)):
            outputs.append(self.cts.output_ct_pts[i].inverse(outputs[i]))

        return torch.cat(outputs, dim=1)
