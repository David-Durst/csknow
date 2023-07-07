from typing import List, Callable

import torch
from torch import nn

from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.order.column_names import team_strs, player_team_str, flatten_list
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, delta_pos_grid_num_cells
from learn_bot.latent.place_area.pos_abs_delta_conversion import compute_new_pos, NavData, \
    delta_one_hot_max_to_index, delta_one_hot_prob_to_index, max_speed_per_half_second
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR, CPU_DEVICE_STR
from learn_bot.libs.positional_encoding import *
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


def range_list_to_index_list(range_list: list[range]) -> list[int]:
    result: list[int] = []
    for range in range_list:
        for i in range:
            result.append(i)
    return result


d2_min = [-2257., -1207., -204.128]
d2_max = [1832., 3157., 236.]


class TransformerNestedHiddenLatentModel(nn.Module):
    internal_width = 512
    cts: IOColumnTransformers
    output_layers: List[nn.Module]
    latent_to_distributions: Callable
    noise_var: float

    def __init__(self, cts: IOColumnTransformers, outer_latent_size: int, inner_latent_size: int, num_layers, num_heads):
        super(TransformerNestedHiddenLatentModel, self).__init__()
        self.cts = cts
        c4_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="c4"))
        #baiting_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="baiting"))
        players_columns = [c4_columns_ranges + #baiting_columns_ranges +
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
        self.num_dim = 3
        self.num_time_steps = len(self.players_pos_columns) // self.num_dim // self.num_players
        assert self.num_players == outer_latent_size
        self.num_players_per_team = self.num_players // 2

        # only different players in same time step to talk to each other
        num_temporal_tokens = self.num_players * self.num_time_steps
        self.temporal_mask_cpu = torch.zeros([num_temporal_tokens, num_temporal_tokens])
        for i in range(num_temporal_tokens):
            for j in range(num_temporal_tokens):
                self.temporal_mask_cpu[i, j] = (i // num_temporal_tokens) != (j // num_temporal_tokens)

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
        self.spatial_positional_encoder, self.spatial_positional_encoder_out_dim = get_embedder()
        self.columns_per_player_time_step = (len(self.players_non_pos_columns) // self.num_players) + \
                                            self.spatial_positional_encoder_out_dim

        # positional encoding library doesn't play well with torchscript, so I'll just make the
        # encoding matrices upfront and add them during inference
        temporal_positional_encoder = PositionalEncoding1D(self.internal_width)
        self.temporal_positional_encoding = \
            temporal_positional_encoder(torch.zeros([self.num_players, self.num_time_steps, self.internal_width]))

        self.embedding_model = nn.Sequential(
            nn.Linear(self.columns_per_player_time_step, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
        )

        self.nav_data_cpu = NavData(CPU_DEVICE_STR)
        self.nav_data_cuda = NavData(CUDA_DEVICE_STR)

        temporal_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.internal_width, nhead=num_heads, batch_first=True)
        self.temporal_transformer_encoder = nn.TransformerEncoder(temporal_transformer_encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
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
        pos_scaled = torch.zeros_like(pos)
        if pos.device.type == CUDA_DEVICE_STR:
            pos_scaled[:, :, 0] = (pos[:, :, 0] - self.d2_min_gpu) / (self.d2_max_gpu - self.d2_min_gpu)
        else:
            pos_scaled[:, :, 0] = (pos[:, :, 0] - self.d2_min_cpu) / (self.d2_max_cpu - self.d2_min_cpu)
        if self.num_time_steps > 1:
            pos_scaled[:, :, 1:] = (pos[:, :, 1:] - max_speed_per_half_second) / (2 * max_speed_per_half_second)
        pos_scaled = torch.clamp(pos_scaled, 0, 1)
        pos_scaled = (pos_scaled * 2) - 1

        if self.noise_var >= 0.:
            means = torch.zeros(pos.shape)
            vars = torch.full(pos.shape, self.noise_var)
            noise = torch.normal(means, vars).to(pos.device.type)
            noise_scaled = noise / (self.d2_max_gpu - self.d2_min_gpu)
            pos_scaled += noise_scaled

        return self.spatial_positional_encoder(pos_scaled)

    def encode_y(self, x_pos, x_non_pos, y, take_max) -> torch.Tensor:
        if take_max:
            y_per_player = delta_one_hot_max_to_index(y)
        else:
            y_per_player = delta_one_hot_prob_to_index(y)
        # shift by 1 so never looking into future (and 0 out for past)
        y_per_player_shifted = torch.roll(y_per_player, 1, dims=1)
        y_per_player_shifted[:, 0] = 0
        if x_pos.device.type == CPU_DEVICE_STR:
            y_pos = compute_new_pos(x_pos, y_per_player_shifted, self.nav_data_cpu)
        else:
            y_pos = compute_new_pos(x_pos, y_per_player_shifted, self.nav_data_cuda)
        y_pos_encoded = self.encode_pos(y_pos)
        #y_pos_time_flattened = rearrange(y_pos_encoded, "b p t d -> b p (t d)")
        y_gathered = torch.cat([y_pos_encoded, x_non_pos], -1)[:, :, 0, :]
        return self.embedding_model(y_gathered)

    def generate_tgt_mask(self, device: str) -> torch.Tensor:
        # base tgt mask that is diagonal to ensure only look at future teammates
        tgt_mask = self.transformer_model.generate_square_subsequent_mask(self.num_players, device)

        # team-based mask
        negs = torch.full((self.num_players, self.num_players), float('-inf'), device=device)
        team_mask = torch.zeros((self.num_players, self.num_players), device=device)
        team_mask[:self.num_players_per_team, self.num_players_per_team:] = \
            negs[:self.num_players_per_team, self.num_players_per_team:]
        team_mask[self.num_players_per_team:, :self.num_players_per_team] = \
            negs[self.num_players_per_team:, :self.num_players_per_team]
        team_mask = tgt_mask + team_mask

        return team_mask

    def forward(self, x, y=None):
        x_pos = rearrange(x[:, self.players_pos_columns], "b (p t d) -> b p t d", p=self.num_players,
                          t=self.num_time_steps, d=self.num_dim)
        if self.num_time_steps > 1:
            x_pos_lagged = torch.roll(x_pos, 1, 2)
            x_pos[:, :, 1:] = x_pos_lagged[:, :, 1:] - x_pos[:, :, 1:]
        x_pos_encoded = self.encode_pos(x_pos)
        x_non_pos = rearrange(x[:, self.players_non_pos_columns], "b (p d) -> b p 1 d", p=self.num_players) \
            .repeat([1, 1, self.num_time_steps, 1])
        x_gathered = torch.cat([x_pos_encoded, x_non_pos], -1)
        x_embedded = self.embedding_model(x_gathered)

        if self.num_time_steps > 1:
            x_batch_player_flattened = rearrange(x_embedded, "b p t d -> (b p) t d")
            batch_temporal_positional_encoding = self.temporal_positional_encoding.repeat([x.shape[0], 1, 1]) \
                .to(x.device.type)
            x_temporal_encoded_batch_player_flattened = x_batch_player_flattened + batch_temporal_positional_encoding
            x_temporal_encoded_player_time_flattened = rearrange(x_temporal_encoded_batch_player_flattened,
                                                                 "(b p) t d -> b (p t) d", p=self.num_players,
                                                                 t=self.num_time_steps)
            x_temporal_embedded_player_time_flattened = \
                self.temporal_transformer_encoder(x_temporal_encoded_player_time_flattened,
                                                  mask=self.temporal_mask_cpu.to(x.device.type))
            x_temporal_embedded = rearrange(x_temporal_embedded_player_time_flattened, "b (p t) d -> b p t d",
                                            p=self.num_players, t=self.num_time_steps)
        else:
            x_temporal_embedded = x_embedded
        x_temporal_embedded_flattened = x_temporal_embedded[:, :, 0, :]


        alive_gathered = x[:, self.alive_columns]
        dead_gathered = alive_gathered < 0.1

        tgt_mask = self.generate_tgt_mask(x.device.type)

        if y is not None:
            y_encoded = self.encode_y(x_pos, x_non_pos, y, True)
            transformed = self.transformer_model(x_temporal_embedded_flattened, y_encoded, tgt_mask=tgt_mask,
                                                 src_key_padding_mask=dead_gathered)#,
                                                 #tgt_key_padding_mask=dead_gathered)
            #transformed = transformed.masked_fill(torch.isnan(transformed), 0)
            latent = self.decoder(transformed)
            return self.logits_output(latent), self.prob_output(latent)
        else:
            y_nested = torch.zeros([x.shape[0], self.num_players, delta_pos_grid_num_cells], device=x.device.type)
            y_nested[:, :, 0] = 1.
            y = rearrange(y_nested, "b p d -> b (p d)")
            memory = self.transformer_model.encoder(x_temporal_embedded_flattened, src_key_padding_mask=dead_gathered)
            for i in range(self.num_players_per_team):
                y_encoded = self.encode_y(x_pos, x_non_pos, y, False)
                transformed = self.transformer_model.decoder(y_encoded, memory, tgt_mask=tgt_mask)
                latent = self.decoder(transformed)
                prob_output_nested = rearrange(self.prob_output(latent), "b (p d) -> b p d", p=self.num_players)
                y_nested[:, i] = prob_output_nested[:, i]
                y_nested[:, i + self.num_players_per_team] = prob_output_nested[:, i + self.num_players_per_team]
            return self.logits_output(latent), self.prob_output(latent), delta_one_hot_prob_to_index(y)
