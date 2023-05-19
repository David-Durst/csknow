from typing import List, Callable

import torch
from torch import nn

from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.order.column_names import team_strs, player_team_str, flatten_list
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR


def range_list_to_index_list(range_list: list[range]) -> list[int]:
    result: list[int] = []
    for range in range_list:
        for i in range:
            result.append(i)
    return result


class TransformerNestedHiddenLatentModel(nn.Module):
    internal_width = 256
    cts: IOColumnTransformers
    output_layers: List[nn.Module]
    latent_to_distributions: Callable
    add_noise: bool

    def __init__(self, cts: IOColumnTransformers, outer_latent_size: int, inner_latent_size: int):
        super(TransformerNestedHiddenLatentModel, self).__init__()
        self.cts = cts
        c4_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="c4"))
        baiting_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="baiting"))
        player_columns = [c4_columns_ranges + baiting_columns_ranges +
                          range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=" " + player_team_str(team_str, player_index)))
                          for team_str in team_strs for player_index in range(max_enemies)]

        self.num_players = len(player_columns)
        assert self.num_players == outer_latent_size
        self.columns_per_players = len(player_columns[0])
        self.player_columns_cpu = torch.tensor(flatten_list(player_columns))
        self.player_columns_gpu = self.player_columns_cpu.to(CUDA_DEVICE_STR)

        alive_columns = [range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=player_place_area_columns.alive))
                         for player_place_area_columns in specific_player_place_area_columns]
        self.alive_columns_cpu = torch.tensor(flatten_list(alive_columns))
        self.alive_columns_gpu = self.alive_columns_cpu.to(CUDA_DEVICE_STR)

        self.pos_columns = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="pos"))
        self.add_noise = False

        self.encoder_model = nn.Sequential(
            nn.Linear(len(player_columns[0]), self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.internal_width, nhead=4, batch_first=True)
        self.transformer_model = nn.TransformerEncoder(transformer_encoder_layer, num_layers=2, enable_nested_tensor=False)

        self.decoder = nn.Sequential(
            nn.Linear(self.internal_width, inner_latent_size),
        )

        self.logits_output = nn.Sequential(
            nn.Flatten(1)
        )

        self.prob_output = nn.Sequential(
            nn.Softmax(dim=2),
            nn.Flatten(1)
        )

    def forward(self, x, noise=None):
        # transform inputs
        x_transformed = self.cts.transform_columns(True, x, x)
        if self.add_noise:
            x_transformed[:, self.pos_columns] += torch.randn([x.shape[0], len(self.pos_columns)]).to(x_transformed.device.type)

        if x_transformed.device.type == CUDA_DEVICE_STR:
            x_gathered = torch.index_select(x_transformed, 1, self.player_columns_gpu)
            alive_gathered = torch.index_select(x_transformed, 1, self.alive_columns_gpu)
        else:
            x_gathered = torch.index_select(x_transformed, 1, self.player_columns_cpu)
            alive_gathered = torch.index_select(x_transformed, 1, self.alive_columns_cpu)

        dead_gathered = alive_gathered < 0.1

        split_x_gathered = x_gathered.unflatten(1, torch.Size([self.num_players, self.columns_per_players]))

        # run model except last layer
        encoded = self.encoder_model(split_x_gathered)

        transformed = self.transformer_model(encoded, src_key_padding_mask=dead_gathered)

        #if torch.isnan(transformed).any():
        #       all_in_tick_dead_gathered = dead_gathered.all(axis=1)
        #       all_in_tick_dead_gathered = all_in_tick_dead_gathered.unflatten(0, [-1, 1, 1]).expand(transformed.shape)
        #       new_transformed = torch.where(all_in_tick_dead_gathered, 0., transformed)
        #       if torch.isnan(new_transformed).any():
        #           print("bad")
        #       transformed = new_transformed

        latent = self.decoder(transformed)

        # https://github.com/pytorch/pytorch/issues/22440 how to parse tuple output
        return self.logits_output(latent), self.prob_output(latent)


#class SimplifiedTransformerNestedHiddenLatentModel(nn.Module):
#
#    def __init__(self):
#        super(SimplifiedTransformerNestedHiddenLatentModel, self).__init__()
#
#        self.transformer_model = nn.Sequential(
#            nn.TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True),
#        )
#
#    def forward(self, x):
#        # transform inputs
#        transformed = self.transformer_model(x)
#
#        return transformed
#
#        ones = torch.rand(encoded.shape).to(CUDA_DEVICE_STR)
#        ones_output = self.simple_transformer_encoder_layer(ones)
#        second_ones_output = self.simple_transformer_encoder_layer(ones)
#        ones_mask_output = self.simple_transformer_encoder_layer(ones, src_key_padding_mask=alive_gathered)
#        ones_mod = torch.clone(ones)
#        ones_mod[:, 1, :] = 2
#        ones_mod_output = self.simple_transformer_encoder_layer(ones_mod)
#        ones_mod_mask_output = self.simple_transformer_encoder_layer(ones_mod, src_key_padding_mask=alive_gathered)
