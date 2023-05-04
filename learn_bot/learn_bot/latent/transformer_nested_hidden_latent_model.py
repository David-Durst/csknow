from typing import List, Callable

import torch
from torch import nn

from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.order.column_names import team_strs, player_team_str, flatten_list
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
        assert self.num_players == outer_latent_size
        self.columns_per_players = len(player_columns[0])
        self.player_columns_cpu = torch.tensor(flatten_list(player_columns))
        self.player_columns_gpu = self.player_columns_cpu.to(CUDA_DEVICE_STR)

        self.encoder_model = nn.Sequential(
            nn.Linear(len(player_columns[0]), self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
        )

        #self.transformer_model = nn.Sequential(
        #    nn.TransformerEncoderLayer(d_model=self.internal_width, nhead=4, batch_first=True, dropout=0.),
        #)

        self.decoder = nn.Sequential(
            nn.Linear(self.internal_width, inner_latent_size),
            nn.Softmax(dim=2),
            nn.Flatten(1)
        )

    def forward(self, x):
        # transform inputs
        x_transformed = self.cts.transform_columns(True, x, x)

        if x_transformed.device.type == CUDA_DEVICE_STR:
            x_gathered = torch.index_select(x_transformed, 1, self.player_columns_gpu)
        else:
            x_gathered = torch.index_select(x_transformed, 1, self.player_columns_cpu)

        split_x_gathered = x_gathered.unflatten(1, torch.Size([self.num_players, self.columns_per_players]))

        # run model except last layer
        encoded = self.encoder_model(split_x_gathered)

        #transformed = self.transformer_model(encoded)

        latent = self.decoder(encoded)

        # https://github.com/pytorch/pytorch/issues/22440 how to parse tuple output
        # hack for now to keep same API, will remove later
        return latent, latent


class SimplifiedTransformerNestedHiddenLatentModel(nn.Module):
    internal_width = 512
    cts: IOColumnTransformers
    output_layers: List[nn.Module]
    latent_to_distributions: Callable

    def __init__(self, cts: IOColumnTransformers, outer_latent_size: int, inner_latent_size: int):
        super(SimplifiedTransformerNestedHiddenLatentModel, self).__init__()
        self.cts = cts
        c4_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="c4"))
        player_columns = [c4_columns_ranges +
                          range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=" " + player_team_str(team_str, player_index)))
                          for team_str in team_strs for player_index in range(max_enemies)]

        self.num_players = len(player_columns)
        assert self.num_players == outer_latent_size
        self.columns_per_players = len(player_columns[0])
        self.player_columns_cpu = torch.tensor(flatten_list(player_columns))
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
        )

        #self.decoder = nn.Sequential(
        #    nn.Linear(self.internal_width, inner_latent_size),
        #    nn.Softmax(dim=2),
        #    nn.Flatten(1)
        #)

    def forward(self, x):
        # transform inputs
        x_transformed = self.cts.transform_columns(True, x, x)

        if x_transformed.device.type == CUDA_DEVICE_STR:
            x_gathered = torch.index_select(x_transformed, 1, self.player_columns_gpu)
        else:
            x_gathered = torch.index_select(x_transformed, 1, self.player_columns_cpu)

        split_x_gathered = x_gathered.unflatten(1, torch.Size([self.num_players, self.columns_per_players]))

        # run model except last layer
        encoded = self.encoder_model(split_x_gathered)

        transformed = self.transformer_model(encoded)

        #latent = self.decoder(transformed)

        return transformed