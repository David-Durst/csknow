from typing import List, Callable

import torch
from torch import nn

from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.order.column_names import team_strs, player_team_str
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR


def range_list_to_index_list(range_list: list[range]) -> list[int]:
    result: list[int] = []
    for range in range_list:
        for i in range:
            result.append(i)
    return result


class TransformerNestedHiddenLatentModel(nn.Module):
    internal_width = 512
    cts: IOColumnTransformers
    output_layers: List[nn.Module]
    latent_to_distributions: Callable

    def __init__(self, cts: IOColumnTransformers, outer_latent_size: int, inner_latent_size: int):
        super(TransformerNestedHiddenLatentModel, self).__init__()
        self.cts = cts
        c4_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="c4"))
        player_columns = [c4_columns_ranges +
                          range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=" " + player_team_str(team_str, player_index)))
                          for team_str in team_strs for player_index in range(max_enemies)]

        self.num_players = len(player_columns)
        self.player_columns_cpu = torch.tensor(player_columns) \
            .unflatten(0, torch.Size([1, len(player_columns)])).expand()
        self.player_columns_gpu = self.player_columns_cpu.to(CUDA_DEVICE_STR)

        self.encoder_model = nn.Sequential(
            nn.Linear(len(player_columns[0]), self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
        )

        self.transformer_model = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=self.internal_width, nhead=4, batch_first=True),
            nn.Linear(self.internal_width, outer_latent_size * inner_latent_size),
            nn.Unflatten(1, torch.Size([outer_latent_size, inner_latent_size])),
            nn.Softmax(dim=2),
            nn.Flatten(1)
        )

    def forward(self, x):
        # transform inputs
        x_transformed = self.cts.transform_columns(True, x, x)

        repeated_x_transformed = x_transformed.unflatten(1, torch.Size([1, x_transformed.shape[-1]])) \
            .expand([-1, self.num_players, -1])

        if x_transformed.device.type == CUDA_DEVICE_STR:
            x_gathered = torch.gather(repeated_x_transformed, 1, self.player_columns_gpu)
        else:
            x_gathered = torch.gather(repeated_x_transformed, 1, self.player_columns_cpu)

        # run model except last layer
        encoded = self.encoder_model(x_gathered)

        latent = self.transformer_model(encoded)

        # https://github.com/pytorch/pytorch/issues/22440 how to parse tuple output
        # hack for now to keep same API, will remove later
        return latent, latent

