import math
from enum import Enum
from typing import List, Callable

import torch
from torch import nn

from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.order.column_names import team_strs, player_team_str, flatten_list
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, num_radial_bins, \
    walking_modifier, ducking_modifier
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import NavData, \
    one_hot_max_to_index, one_hot_prob_to_index, max_speed_per_second
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR, CPU_DEVICE_STR
from learn_bot.libs.positional_encoding import *
from einops import rearrange, repeat
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


def range_list_to_index_list(range_list: list[range]) -> list[int]:
    result: list[int] = []
    for range in range_list:
        for i in range:
            result.append(i)
    return result


def get_player_columns_by_str(cts: IOColumnTransformers, contained_str: str) -> List[int]:
    return flatten_list(
        [range_list_to_index_list(
            cts.get_name_ranges(True, False, contained_str=contained_str + " " + player_team_str(team_str, player_index)))
            for team_str in team_strs for player_index in range(max_enemies)]
    )


d2_min = [-2257., -1207., -204.128]
d2_max = [1832., 3157., 236.]
stature_to_speed_list = [max_speed_per_second, max_speed_per_second * walking_modifier,
                         max_speed_per_second * ducking_modifier]


class PlayerMaskType(Enum):
    NoMask = 1
    EveryoneTemporalOnlyMask = 2
    EveryoneFullMask = 3

    def __str__(self) -> str:
        if self == PlayerMaskType.NoMask:
            return "NoMask"
        elif self == PlayerMaskType.EveryoneTemporalOnlyMask:
            return "EveryoneTemporalOnlyMask"
        else:
            return "EveryoneFullMask"


class TransformerNestedHiddenLatentModel(nn.Module):
    internal_width = 128
    cts: IOColumnTransformers
    output_layers: List[nn.Module]
    latent_to_distributions: Callable
    noise_var: float

    def __init__(self, cts: IOColumnTransformers, num_players: int, num_output_time_steps, num_radial_bins: int,
                 num_layers: int, num_heads: int, player_mask_type: PlayerMaskType):
        super(TransformerNestedHiddenLatentModel, self).__init__()
        self.cts = cts
        # transformed/transformed doesn't matter since no angles and all categorical variables are
        # "distirbution" type meaning pre one hot encoded
        c4_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="c4"))
        #baiting_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="baiting"))
        players_columns = [c4_columns_ranges + #baiting_columns_ranges +
                          range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=" " + player_team_str(team_str, player_index)))
                          for team_str in team_strs for player_index in range(max_enemies)]
        self.players_pos_columns = get_player_columns_by_str(cts, "player pos")
        self.players_nearest_crosshair_to_enemy_columns = \
            get_player_columns_by_str(cts, "player nearest crosshair distance to enemy")
        #self.players_hurt_in_last_5s_columns = get_player_columns_by_str(cts, "player hurt in last 5s")
        #self.players_fire_in_last_5s_columns = get_player_columns_by_str(cts, "player fire in last 5s")
        #self.players_enemy_visible_in_last_5s_columns = get_player_columns_by_str(cts, "player enemy visible in last 5s")
        #self.players_health_columns = get_player_columns_by_str(cts, "player health")
        #self.players_armor_columns = get_player_columns_by_str(cts, "player armor")
        #self.players_vel_columns = flatten_list(
        #    [range_list_to_index_list(cts.get_name_ranges(True, False, contained_str="player velocity " + player_team_str(team_str, player_index)))
        #     for team_str in team_strs for player_index in range(max_enemies)]
        #)
        self.player_temporal_columns = self.players_pos_columns + self.players_nearest_crosshair_to_enemy_columns
        self.players_non_temporal_columns = flatten_list([
            [player_column for player_column in player_columns if player_column not in self.player_temporal_columns]
            for player_columns in players_columns
        ])
        self.num_similarity_columns = 2

        self.num_players = len(players_columns)
        self.num_dim = 3
        self.num_input_time_steps = len(self.players_pos_columns) // self.num_dim // self.num_players
        self.num_output_time_steps = num_output_time_steps
        assert self.num_output_time_steps <= self.num_input_time_steps
        assert self.num_players == num_players
        self.num_players_per_team = self.num_players // 2

        # only different players in same time step to talk to each other
        def build_per_player_mask(per_player_mask: torch.Tensor, num_temporal_tokens: int, num_time_steps: int):
            for i in range(num_temporal_tokens):
                for j in range(num_temporal_tokens):
                    per_player_mask[i, j] = (i // num_time_steps) != (j // num_time_steps)

        num_input_temporal_tokens = self.num_players * self.num_input_time_steps
        self.input_per_player_mask_cpu = torch.zeros([num_input_temporal_tokens, num_input_temporal_tokens],
                                                     dtype=torch.bool)
        if player_mask_type != PlayerMaskType.NoMask:
            build_per_player_mask(self.input_per_player_mask_cpu, num_input_temporal_tokens, self.num_input_time_steps)

        num_output_temporal_tokens = self.num_players * self.num_output_time_steps
        self.output_per_player_mask_cpu = torch.zeros([num_output_temporal_tokens, num_output_temporal_tokens],
                                                      dtype=torch.bool)
        if player_mask_type != PlayerMaskType.NoMask:
            build_per_player_mask(self.output_per_player_mask_cpu, num_output_temporal_tokens, self.num_output_time_steps)

        self.player_mask_type = player_mask_type

        self.alive_columns = flatten_list(
            [range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=player_place_area_columns.alive))
             for player_place_area_columns in specific_player_place_area_columns]
        )

        self.noise_var = -1.

        self.d2_min_cpu = torch.tensor(d2_min)
        self.d2_max_cpu = torch.tensor(d2_max)
        self.d2_min_gpu = self.d2_min_cpu.to(CUDA_DEVICE_STR)
        self.d2_max_gpu = self.d2_max_cpu.to(CUDA_DEVICE_STR)

        # radial speed matrices
        self.stature_to_speed_cpu = torch.tensor(stature_to_speed_list)
        self.stature_to_speed_gpu = self.stature_to_speed_cpu.to(CUDA_DEVICE_STR)


        # NERF code calls it positional embedder, but it's encoder since not learned
        self.spatial_positional_encoder, self.spatial_positional_encoder_out_dim = get_embedder()
        self.columns_per_player_time_step = (len(self.players_non_temporal_columns) // self.num_players) + \
                                            self.num_similarity_columns + \
                                            self.spatial_positional_encoder_out_dim + \
                                            len(self.players_nearest_crosshair_to_enemy_columns) // self.num_players // self.num_input_time_steps # + \
                                            #(len(self.players_vel_columns) // self.num_players // self.num_time_steps)

        # positional encoding library doesn't play well with torchscript, so I'll just make the
        # encoding matrices upfront and add them during inference
        temporal_positional_encoder = PositionalEncoding1D(self.internal_width)
        self.temporal_positional_encoding = \
            temporal_positional_encoder(torch.zeros([self.num_players, self.num_input_time_steps, self.internal_width]))

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
            nn.Linear(self.internal_width, num_radial_bins),
        )

        # dim 0 is batch, 1 is players, 2 is output time step, 3 is probabilities/logits
        #self.logits_output = nn.Sequential(
        #    nn.Flatten(1)
        #)

        self.prob_output = nn.Sequential(
            nn.Softmax(dim=3),
        )

    def encode_pos(self, pos: torch.tensor, enable_noise=True):
        # https://arxiv.org/pdf/2003.08934.pdf
        # 5.1 - everything is normalized -1 to 1
        pos_scaled = torch.zeros_like(pos)
        if pos.device.type == CUDA_DEVICE_STR:
            pos_scaled[:, :, 0] = (pos[:, :, 0] - self.d2_min_gpu) / (self.d2_max_gpu - self.d2_min_gpu)
        else:
            pos_scaled[:, :, 0] = (pos[:, :, 0] - self.d2_min_cpu) / (self.d2_max_cpu - self.d2_min_cpu)
        if self.num_input_time_steps > 1:
            # this may divide by a little too much (if time between pos < 1s, but future proof for later with variable
            # tick rate inputs)
            pos_scaled[:, :, 1:] = (pos[:, :, 1:] - 130) / (130 * 2)
        pos_scaled = torch.clamp(pos_scaled, 0, 1)
        pos_scaled = (pos_scaled * 2) - 1

        if self.noise_var >= 0. and enable_noise:
            means = torch.zeros(pos.shape)
            vars = torch.full(pos.shape, self.noise_var)
            noise = torch.normal(means, vars).to(pos.device.type)
            noise_scaled = noise / (self.d2_max_gpu - self.d2_min_gpu)
            pos_scaled += noise_scaled

        return self.spatial_positional_encoder(pos_scaled)

    #def encode_y(self, x_pos, x_non_pos, y, take_max) -> torch.Tensor:
    #    if take_max:
    #        y_per_player = one_hot_max_to_index(y)
    #    else:
    #        y_per_player = one_hot_prob_to_index(y)
    #    # shift by 1 so never looking into future (and 0 out for past)
    #    y_per_player_shifted = torch.roll(y_per_player, 1, dims=1)
    #    y_per_player_shifted[:, 0] = 0
    #    #return self.y_embedding_model(rearrange(y_per_player_shifted, "b y -> b y 1"))
    #    #x_pos_shifted = torch.roll(x_pos, 1, dims=1)
    #    #x_pos_zeros = torch.zeros_like(x_pos)
    #    #if x_pos.device.type == CPU_DEVICE_STR:
    #    #    y_pos = compute_new_pos(x_pos, y_per_player_shifted, self.nav_data_cpu, False, self.stature_to_speed_cpu)
    #    #else:
    #    #    y_pos = compute_new_pos(x_pos, y_per_player_shifted, self.nav_data_cuda, False, self.stature_to_speed_gpu)
    #    #y_pos2 = compute_new_pos(x_pos_shifted, y_per_player_shifted, self.nav_data_cuda)
    #    y_pos_encoded = self.encode_pos(x_pos, enable_noise=False)
    #    #y_pos_time_flattened = rearrange(y_pos_encoded, "b p t d -> b p (t d)")
    #    y_gathered = torch.cat([y_pos_encoded, x_non_pos], -1)[:, :, 0, :]
    #    return self.embedding_model(y_gathered)

    def generate_tgt_mask(self, device: str) -> torch.Tensor:
        num_player_time_steps = self.num_players * self.num_output_time_steps
        # base tgt mask that is diagonal to ensure only look at future teammates
        tgt_mask = self.transformer_model.generate_square_subsequent_mask(num_player_time_steps, device)

        # team-based mask
        negs = torch.full((num_player_time_steps, num_player_time_steps), float('-inf'), device=device)
        team_mask = torch.zeros((num_player_time_steps, num_player_time_steps), device=device)
        num_player_time_steps_per_team = self.num_players_per_team * self.num_output_time_steps
        team_mask[:num_player_time_steps_per_team, num_player_time_steps_per_team:] = \
            negs[:num_player_time_steps_per_team, num_player_time_steps_per_team:]
        team_mask[num_player_time_steps_per_team:, :num_player_time_steps_per_team] = \
            negs[num_player_time_steps_per_team:, :num_player_time_steps_per_team]
        team_mask = tgt_mask + team_mask

        return team_mask

    def forward(self, x, similarity, temperature):
        if self.num_input_time_steps < 2:
            raise Exception("must have history")

        x_pos = rearrange(x[:, self.players_pos_columns], "b (p t d) -> b p t d", p=self.num_players,
                          t=self.num_input_time_steps, d=self.num_dim)
        x_crosshair = rearrange(x[:, self.players_nearest_crosshair_to_enemy_columns], "b (p t) -> b p t 1",
                                p=self.num_players, t=self.num_input_time_steps)
        # delta encode prior pos
        x_pos_lagged = torch.roll(x_pos, 1, 2)
        x_pos[:, :, 1:] = x_pos_lagged[:, :, 1:] - x_pos[:, :, 1:]
        x_pos_encoded = self.encode_pos(x_pos)

        #x_vel = rearrange(x[:, self.players_vel_columns], "b (p t d) -> b p t d", p=self.num_players,
        #                  t=self.num_time_steps, d=self.num_dim)
        #x_vel_scaled = x_vel / max_speed_per_second


        x_non_pos_without_similarity = \
            rearrange(x[:, self.players_non_temporal_columns], "b (p d) -> b p 1 d", p=self.num_players) \
            .repeat([1, 1, self.num_input_time_steps, 1])
        similarity_expanded = rearrange(similarity, '(b p t) d -> b p t d', p=1, t=1) \
            .repeat([1, self.num_players, self.num_input_time_steps, 1])
        x_non_pos = torch.concat([x_non_pos_without_similarity, similarity_expanded], dim=-1)
        x_gathered = torch.cat([x_pos_encoded, x_crosshair, x_non_pos], -1)
        #x_gathered = torch.cat([x_pos_encoded, x_vel_scaled, x_non_pos], -1)
        x_embedded = self.embedding_model(x_gathered)

        # apply temporal encoder to get one token per player
        # need to flatten first across batch/player to do temporal positional encoding per player
        x_batch_player_flattened = rearrange(x_embedded, "b p t d -> (b p) t d")
        batch_temporal_positional_encoding = self.temporal_positional_encoding.repeat([x.shape[0], 1, 1]) \
            .to(x.device.type)
        x_temporal_encoded_batch_player_flattened = x_batch_player_flattened + batch_temporal_positional_encoding
        # next flatten all tokens into on sequence per batch. using temporal_mask to ensure only comparing tokens
        # from same players
        x_temporal_encoded_player_time_flattened = rearrange(x_temporal_encoded_batch_player_flattened,
                                                             "(b p) t d -> b (p t) d", p=self.num_players,
                                                             t=self.num_input_time_steps)
        x_temporal_embedded_player_time_flattened = \
            self.temporal_transformer_encoder(x_temporal_encoded_player_time_flattened,
                                              mask=self.input_per_player_mask_cpu.to(x.device.type))
        # take last token per player
        x_temporal_embedded = rearrange(x_temporal_embedded_player_time_flattened, "b (p t) d -> b p t d",
                                        p=self.num_players, t=self.num_input_time_steps)
        x_temporal_embedded_flattened = rearrange(x_temporal_embedded[:, :, 0:self.num_output_time_steps, :],
                                                  "b p t d -> b (p t) d")


        alive_gathered = x[:, self.alive_columns]
        alive_gathered_temporal = repeat(alive_gathered, 'b p -> b (p repeat)', repeat=self.num_output_time_steps)
            #'b p i -> b (p i)')
        dead_gathered = alive_gathered_temporal < 0.1

        tgt_mask = self.generate_tgt_mask(x.device.type)

        x_pos_encoded_no_noise = self.encode_pos(x_pos, enable_noise=False)
        x_gathered_no_noise = rearrange(
            torch.cat([x_pos_encoded_no_noise, x_crosshair, x_non_pos], -1)[:, :, 0:self.num_output_time_steps, :],
            "b p t d -> b (p t) d")
        x_embedded_no_noise = self.embedding_model(x_gathered_no_noise)
        if self.player_mask_type == PlayerMaskType.EveryoneFullMask:
            output_per_player_mask = self.output_per_player_mask_cpu.to(x.device.type)
            tgt_mask = tgt_mask + (torch.where(output_per_player_mask, -1. * math.inf, 0.))
            transformed = self.transformer_model(x_temporal_embedded_flattened, x_embedded_no_noise, tgt_mask=tgt_mask,
                                                 src_mask=output_per_player_mask)
        else:
            transformed = self.transformer_model(x_temporal_embedded_flattened, x_embedded_no_noise, tgt_mask=tgt_mask,
                                                 src_key_padding_mask=dead_gathered)
        transformed_nested = rearrange(transformed, "b (p t) d -> b p t d",
                                       p=self.num_players, t=self.num_output_time_steps)
        latent = self.decoder(transformed_nested)
        prob_output = self.prob_output(latent / temperature)
        return latent, prob_output, one_hot_prob_to_index(prob_output)
        #if y is not None:
        #    y_encoded = self.encode_y(x_pos, x_non_pos, y, True)
        #    transformed = self.transformer_model(x_temporal_embedded_flattened, y_encoded, tgt_mask=tgt_mask,
        #                                         src_key_padding_mask=dead_gathered)#,
        #                                         #tgt_key_padding_mask=dead_gathered)
        #    #transformed = transformed.masked_fill(torch.isnan(transformed), 0)
        #    latent = self.decoder(transformed)
        #    return self.logits_output(latent), self.prob_output(latent)
        #else:
        #    y_nested = torch.zeros([x.shape[0], self.num_players, num_radial_bins], device=x.device.type)
        #    y_nested[:, :, 0] = 1.
        #    y = rearrange(y_nested, "b p d -> b (p d)")
        #    memory = self.transformer_model.encoder(x_temporal_embedded_flattened, src_key_padding_mask=dead_gathered)
        #    for i in range(self.num_players_per_team):
        #        y_encoded = self.encode_y(x_pos, x_non_pos, y, False)
        #        transformed = self.transformer_model.decoder(y_encoded, memory, tgt_mask=tgt_mask)
        #        latent = self.decoder(transformed)
        #        prob_output_nested = rearrange(self.prob_output(latent), "b (p d) -> b p d", p=self.num_players)
        #        y_nested[:, i] = prob_output_nested[:, i]
        #        y_nested[:, i + self.num_players_per_team] = prob_output_nested[:, i + self.num_players_per_team]
        #    return self.logits_output(latent), self.prob_output(latent), one_hot_prob_to_index(y)
