from typing import List, Callable

import torch
from torch import nn

from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.order.column_names import team_strs, player_team_str, flatten_list
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.pos_abs_delta_conversion import compute_new_pos, load_nav_region_and_above_below, \
    delta_one_hot_to_index
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR
from learn_bot.libs.positional_encoding import *
from einops import rearrange


def range_list_to_index_list(range_list: list[range]) -> list[int]:
    result: list[int] = []
    for range in range_list:
        for i in range:
            result.append(i)
    return result


d2_min = [-2257., -1207., -204.128]
d2_max = [1832., 3157., 236.]


class TransformerNestedHiddenLatentModel(nn.Module):
    internal_width = 256
    cts: IOColumnTransformers
    output_layers: List[nn.Module]
    latent_to_distributions: Callable
    noise_var: float

    def __init__(self, cts: IOColumnTransformers, outer_latent_size: int, inner_latent_size: int, num_layers, num_heads):
        super(TransformerNestedHiddenLatentModel, self).__init__()
        self.cts = cts
        c4_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="c4"))
        baiting_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="baiting"))
        players_columns = [c4_columns_ranges + baiting_columns_ranges +
                          range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=" " + player_team_str(team_str, player_index)))
                          for team_str in team_strs for player_index in range(max_enemies)]
        self.players_pos_columns = flatten_list(
            [range_list_to_index_list(cts.get_name_ranges(True, False, contained_str="player pos " + player_team_str(team_str, player_index)))
             for team_str in team_strs for player_index in range(max_enemies)]
        )
        self.players_non_pos_columns = flatten_list([
            [player_column for player_column in player_columns if player_column not in self.players_pos_columns]
            for player_columns in players_columns
        ])

        self.num_players = len(players_columns)
        assert self.num_players == outer_latent_size

        self.alive_columns = flatten_list(
            [range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=player_place_area_columns.alive))
             for player_place_area_columns in specific_player_place_area_columns]
        )

        self.noise_var = -1.

        self.d2_min_cpu = torch.tensor(d2_min)
        self.d2_max_cpu = torch.tensor(d2_max)
        self.d2_min_gpu = self.d2_min_cpu.to(CUDA_DEVICE_STR)
        self.d2_max_gpu = self.d2_max_cpu.to(CUDA_DEVICE_STR)

        # NERF code calls it positional embedder, but it's encoder since not learned
        self.positional_encoder, self.positional_encoder_out_dim = get_embedder()
        self.columns_per_player = (len(self.players_non_pos_columns) // self.num_players) + self.positional_encoder_out_dim

        self.encoder_model = nn.Sequential(
            nn.Linear(self.columns_per_player, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
        )

        self.nav_region, self.nav_above_below = load_nav_region_and_above_below()

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.internal_width, nhead=num_heads, batch_first=True)
        transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.transformer_model = nn.Transformer(d_model=self.internal_width, nhead=num_heads,
                                                num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                                custom_encoder=transformer_encoder, batch_first=True)

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

    def encode_pos(self, pos: torch.tensor):
        # https://arxiv.org/pdf/2003.08934.pdf
        # 5.1 - everything is normalized -1 to 1
        if pos.device.type == CUDA_DEVICE_STR:
            pos_scaled = (pos - self.d2_min_gpu) / (self.d2_max_gpu - self.d2_min_gpu)
        else:
            pos_scaled = (pos - self.d2_min_cpu) / (self.d2_max_cpu - self.d2_min_cpu)
        pos_scaled = torch.clamp(pos_scaled, 0, 1)
        pos_scaled = (pos_scaled * 2) - 1

        if self.noise_var >= 0.:
            means = torch.zeros(pos.shape)
            vars = torch.full(pos.shape, self.noise_var)
            noise = torch.normal(means, vars).to(pos.device.type)
            noise_scaled = noise / (self.d2_max_gpu - self.d2_min_gpu)
            pos_scaled += noise_scaled

        return self.positional_encoder(pos_scaled)

    #https://pytorch.org/tutorials/beginner/translation_transformer.html
    def generate_square_subsequent_mask(self, device):
        mask = (torch.triu(torch.ones((self.num_players, self.num_players), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, y=None):
        # transform inputs
        #x_transformed = self.cts.transform_columns(True, x, x)

        x_pos = rearrange(x[:, self.players_pos_columns], "b (p d) -> b p d", p=self.num_players, d=3)

        x_pos_encoded = self.encode_pos(x_pos)

        x_non_pos = rearrange(x[:, self.players_non_pos_columns], "b (p d) -> b p d", p=self.num_players)
        x_gathered = torch.cat([x_pos_encoded, x_non_pos], -1)

        alive_gathered = x[:, self.alive_columns]
        dead_gathered = alive_gathered < 0.1

        # run model except last layer
        x_encoded = self.encoder_model(x_gathered)

        if y is not None:
            y_per_player = delta_one_hot_to_index(y)
            # shift by 1 so never looking into future (and 0 out for past)
            y_per_player_shifted = torch.roll(y_per_player, 1, dims=1)
            y_per_player_shifted[:, 0] = 0
            y_pos = rearrange(compute_new_pos(x_pos, y_per_player_shifted, self.nav_above_below, self.nav_region),
                              "b (p d) -> b p d", p=self.num_players, d=3)
            y_pos_encoded = self.encode_pos(y_pos)
            y_gathered = torch.cat([y_pos_encoded, x_non_pos], -1)
            y_encoded = self.encoder_model(y_gathered)

        tgt_mask = self.generate_square_subsequent_mask(x.device.type)
        transformed = self.transformer_model(x_encoded, x_encoded if y is None else y_encoded, tgt_mask=tgt_mask, src_key_padding_mask=dead_gathered)
        #transformed = self.transformer_model(x_encoded, y_encoded, src_key_padding_mask=dead_gathered)

        #if torch.isnan(transformed).any():
        #       all_in_tick_dead_gathered = dead_gathered.all(axis=1)
        #       all_in_tick_dead_gathered = all_in_tick_dead_gathered.unflatten(0, [-1, 1, 1]).expand(transformed.shape)
        #       new_transformed = torch.where(all_in_tick_dead_gathered, 0., transformed)
        #       if torch.isnan(new_transformed).any():
        #           print("bad")
        #       transformed = new_transformed

        latent = self.decoder(transformed)
        #max_diff = -1
        #for i in range(transformed.shape[0]):
        #    for j in range(transformed.shape[1]):
        #        tmp_l = self.decoder(transformed[i, j])
        #        max_diff = max(torch.max(torch.abs(latent[i, j] - tmp_l)), max_diff)


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
