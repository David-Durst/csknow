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
            nn.Linear(cts.input_ct.get_feature_names_out().size, len(self.internal_width))
        )

        for output_range in cts.get_output_name_ranges():
            self.output_layers.append(nn.Linear(self.internal_width, len(output_range)))

    def forward(self, x):
        logits = self.inner_model(x)
        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(logits))
        return torch.cat(outputs, dim=1)
