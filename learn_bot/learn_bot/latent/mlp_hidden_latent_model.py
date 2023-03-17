from typing import List, Callable
from torch import nn
from learn_bot.libs.io_transforms import IOColumnTransformers


class MLPHiddenLatentModel(nn.Module):
    internal_width = 1024
    cts: IOColumnTransformers
    output_layers: List[nn.Module]
    latent_to_distributions: Callable

    def __init__(self, cts: IOColumnTransformers, latent_size: int, latent_to_distributions: Callable):
        super(MLPHiddenLatentModel, self).__init__()
        self.cts = cts
        self.inner_model = nn.Sequential(
            nn.Linear(cts.get_name_ranges(True, True)[-1].stop, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, latent_size),
            nn.Softmax(dim=1)
        )
        self.latent_to_distributions = latent_to_distributions

    def forward(self, x):
        # transform inputs
        x_transformed = self.cts.transform_columns(True, x, x)

        # run model except last layer
        latent = self.inner_model(x_transformed)

        # https://github.com/pytorch/pytorch/issues/22440 how to parse tuple output
        return latent, self.latent_to_distributions(latent)

