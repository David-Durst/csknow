import math
from enum import Enum
from typing import List, Callable

import torch
from torch import nn

from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.order.column_names import team_strs, player_team_str, flatten_list, all_prior_and_cur_ticks
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, num_radial_bins, \
    walking_modifier, ducking_modifier
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import NavData, \
    one_hot_max_to_index, one_hot_prob_to_index, max_speed_per_second, max_run_speed_per_sim_tick
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
    TeammateTemporalOnlyMask = 4
    TeammateFullMask = 5

    def __str__(self) -> str:
        if self == PlayerMaskType.NoMask:
            return "NoMask"
        elif self == PlayerMaskType.EveryoneTemporalOnlyMask:
            return "EveryoneTemporalOnlyMask"
        elif self == PlayerMaskType.EveryoneFullMask:
            return "EveryoneFullMask"
        elif self == PlayerMaskType.TeammateTemporalOnlyMask:
            return "TeammateTemporalOnlyMask"
        else:
            return "TeammateFullMask"


class TransformerNestedHiddenLatentModel(nn.Module):
    internal_width: int
    cts: IOColumnTransformers
    output_layers: List[nn.Module]
    latent_to_distributions: Callable
    noise_var: float

    def __init__(self, cts: IOColumnTransformers, internal_width: int, num_players: int, num_input_time_steps: int,
                 num_output_time_steps: int, num_radial_bins: int,
                 num_layers: int, num_heads: int, player_mask_type: PlayerMaskType):
        super(TransformerNestedHiddenLatentModel, self).__init__()
        self.cts = cts
        self.internal_width = internal_width
        # transformed/transformed doesn't matter since no angles and all categorical variables are
        # "distirbution" type meaning pre one hot encoded
        all_c4_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="c4"))
        #_, c4_columns_names = cts.get_name_ranges(True, True, contained_str="c4", include_names=True)
        c4_decrease_distance_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="decrease distance to c4"))
        non_player_c4_columns_ranges = [i for i in all_c4_columns_ranges if i not in c4_decrease_distance_columns_ranges]
        #baiting_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="baiting"))
        players_columns = [non_player_c4_columns_ranges + #baiting_columns_ranges +
                          range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=" " + player_team_str(team_str, player_index)))
                          for team_str in team_strs for player_index in range(max_enemies)]
        #players_columns_names = [cts.get_name_ranges(True, True, contained_str=" " + player_team_str(team_str, player_index), include_names=True)[1]
        #                         for team_str in team_strs for player_index in range(max_enemies)]
        self.num_players = len(players_columns)
        self.num_dim = 3
        self.num_heads = num_heads

        all_players_pos_columns = get_player_columns_by_str(cts, "player pos")
        all_players_pos_columns_tensor = torch.IntTensor(all_players_pos_columns)
        nested_players_pos_columns_tensor = rearrange(all_players_pos_columns_tensor, '(p t d) -> p t d',
                                                      p=self.num_players, t=all_prior_and_cur_ticks, d=self.num_dim)
        self.players_pos_columns = rearrange(nested_players_pos_columns_tensor[:, 0:num_input_time_steps, :],
                                             'p t d -> (p t d)',
                                             p=self.num_players, t=num_input_time_steps, d=self.num_dim).tolist()
        self.players_cur_pos_columns = rearrange(nested_players_pos_columns_tensor[:, 0, :], 'p d -> (p d)',
                                                 p=self.num_players, d=self.num_dim).tolist()
        all_players_nearest_crosshair_to_enemy_columns = \
            get_player_columns_by_str(cts, "player nearest crosshair distance to enemy")
        all_players_nearest_crosshair_to_enemy_columns_tensor = \
            torch.IntTensor(all_players_nearest_crosshair_to_enemy_columns)
        nested_players_nearest_crosshair_to_enemy_columns_tensor = \
            rearrange(all_players_nearest_crosshair_to_enemy_columns_tensor, '(p t) -> p t', p=self.num_players,
                      t=all_prior_and_cur_ticks)
        self.players_nearest_crosshair_to_enemy_columns = \
            rearrange(nested_players_nearest_crosshair_to_enemy_columns_tensor[:, 0:num_input_time_steps],
                      'p t -> (p t)', p=self.num_players, t=num_input_time_steps).tolist()

        #self.players_hurt_in_last_5s_columns = get_player_columns_by_str(cts, "player hurt in last 5s")
        #self.players_fire_in_last_5s_columns = get_player_columns_by_str(cts, "player fire in last 5s")
        #self.players_enemy_visible_in_last_5s_columns = get_player_columns_by_str(cts, "player enemy visible in last 5s")
        #self.players_health_columns = get_player_columns_by_str(cts, "player health")
        #self.players_armor_columns = get_player_columns_by_str(cts, "player armor")
        #self.players_vel_columns = flatten_list(
        #    [range_list_to_index_list(cts.get_name_ranges(True, False, contained_str="player velocity " + player_team_str(team_str, player_index)))
        #     for team_str in team_strs for player_index in range(max_enemies)]
        #)
        self.player_all_temporal_columns = all_players_pos_columns + all_players_nearest_crosshair_to_enemy_columns
        self.players_non_temporal_columns = flatten_list([
            [player_column for player_column in player_columns if player_column not in self.player_all_temporal_columns]
            for player_columns in players_columns
        ])
        self.num_similarity_columns = 2

        # may have more input time steps loaded than used
        self.total_input_time_steps = len(all_players_pos_columns) // self.num_dim // self.num_players
        assert all_prior_and_cur_ticks == self.total_input_time_steps
        self.num_input_time_steps = num_input_time_steps
        assert self.num_input_time_steps <= self.total_input_time_steps
        self.num_output_time_steps = num_output_time_steps
        assert self.num_output_time_steps <= self.num_input_time_steps or self.num_input_time_steps == 1
        assert self.num_players == num_players
        self.num_players_per_team = self.num_players // 2

        # only different players in same time step to talk to each other
        def build_per_player_mask(per_player_mask: torch.Tensor, num_temporal_tokens: int, num_time_steps: int,
                                  teammate_only_mask: bool):
            for i in range(num_temporal_tokens):
                for j in range(num_temporal_tokens):
                    if teammate_only_mask:
                        per_player_mask[i, j] = (i // num_time_steps) != (j // num_time_steps) and \
                                                (i // num_time_steps // self.num_players_per_team) == \
                                                (j // num_time_steps // self.num_players_per_team)
                    else:
                        per_player_mask[i, j] = (i // num_time_steps) != (j // num_time_steps)

        num_input_temporal_tokens = self.num_players * self.num_input_time_steps
        self.input_per_player_mask_cpu = torch.zeros([num_input_temporal_tokens, num_input_temporal_tokens],
                                                     dtype=torch.bool)
        self.input_per_player_no_history_mask_cpu = torch.zeros([self.num_players, self.num_players],
                                                                dtype=torch.bool)
        teammate_only_mask = (player_mask_type == PlayerMaskType.TeammateTemporalOnlyMask) or \
                             (player_mask_type == PlayerMaskType.TeammateFullMask)
        if player_mask_type != PlayerMaskType.NoMask:
            build_per_player_mask(self.input_per_player_mask_cpu, num_input_temporal_tokens, self.num_input_time_steps,
                                  teammate_only_mask)
            build_per_player_mask(self.input_per_player_no_history_mask_cpu, self.num_players, 1, teammate_only_mask)

        num_output_temporal_tokens = self.num_players * self.num_output_time_steps
        self.output_per_player_mask_cpu = torch.zeros([num_output_temporal_tokens, num_output_temporal_tokens],
                                                      dtype=torch.bool)
        if player_mask_type != PlayerMaskType.NoMask:
            build_per_player_mask(self.output_per_player_mask_cpu, num_output_temporal_tokens, self.num_output_time_steps,
                                  teammate_only_mask)

        self.player_mask_type = player_mask_type

        self.alive_columns = flatten_list(
            [range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=player_place_area_columns.alive))
             for player_place_area_columns in specific_player_place_area_columns]
        )

        self.noise_var = -1.
        self.drop_history = False

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
                                            1 # for the nearest_crosshair_to_enemy_columns instead of next line
                                            #len(self.players_nearest_crosshair_to_enemy_columns) // self.num_players // self.num_input_time_steps # + \
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
        self.spatial_transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        #self.transformer_model = nn.Transformer(d_model=self.internal_width, nhead=num_heads,
        #                                        num_encoder_layers=num_layers, num_decoder_layers=num_layers,
        #                                        custom_encoder=transformer_encoder, batch_first=True)

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
            # + rather than - here as speed is positive, d2 dimensions are negative
            # multiply by 2 so be sure to capture outliers
            pos_scaled[:, :, 1:] = (pos[:, :, 1:] + (2 * max_run_speed_per_sim_tick)) / (4 * max_run_speed_per_sim_tick)
        pos_scaled = torch.clamp(pos_scaled, 0, 1)
        pos_scaled = (pos_scaled * 2) - 1

        if self.noise_var >= 0. and enable_noise:
            means = torch.zeros(pos.shape)
            vars = torch.full(pos.shape, self.noise_var)
            noise = torch.normal(means, vars).to(pos.device.type)
            noise_scaled = noise / (self.d2_max_gpu - self.d2_min_gpu)
            pos_scaled += noise_scaled

        return self.spatial_positional_encoder(pos_scaled)

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
        x_pos = rearrange(x[:, self.players_pos_columns], "b (p t d) -> b p t d", p=self.num_players,
                          t=self.num_input_time_steps, d=self.num_dim)
        x_crosshair = rearrange(x[:, self.players_nearest_crosshair_to_enemy_columns], "b (p t) -> b p t 1",
                                p=self.num_players, t=self.num_input_time_steps)
        # delta encode prior pos
        #if self.num_input_time_steps < self.total_input_time_steps:
        #    x_pos = x_pos_all_time_steps[:, :, 0:self.num_input_time_steps, :]
        #    x_crosshair = x_crosshair_all_time_steps[:, :, 0:self.num_input_time_steps, :]
        #else:
        #    x_pos = x_pos_all_time_steps
        #    x_crosshair = x_crosshair_all_time_steps
        if self.num_input_time_steps > 1:
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

        # figure out alive and dead for masking out dead ploayers
        alive_gathered = x[:, self.alive_columns]
        alive_gathered_input_temporal = repeat(alive_gathered, 'b p -> b (p repeat)',
                                               repeat=self.num_input_time_steps)
        dead_gathered_input = alive_gathered_input_temporal < 0.1
        alive_gathered_output_temporal = repeat(alive_gathered, 'b p -> b (p repeat)',
                                                repeat=self.num_output_time_steps)
        dead_gathered_output = alive_gathered_output_temporal < 0.1

        # apply temporal encoder to get one token per player
        # need to flatten first across batch/player to do temporal positional encoding per player
        x_batch_player_flattened = rearrange(x_embedded, "b p t d -> (b p) t d")
        batch_temporal_positional_encoding = self.temporal_positional_encoding.repeat([x.shape[0], 1, 1]) \
            .to(x.device.type)
        x_temporal_encoded_batch_player_flattened = x_batch_player_flattened + batch_temporal_positional_encoding
        if self.drop_history:
            x_temporal_encoded_player_time_flattened = rearrange(x_temporal_encoded_batch_player_flattened[:, 0, :],
                                                                 "(b p) d -> b p d", p=self.num_players)
            combined_input_mask = combine_padding_sequence_masks(
                self.input_per_player_no_history_mask_cpu.to(x.device.type), dead_gathered_input, self.num_heads)
            x_temporal_embedded_player_time_flattened = \
                self.temporal_transformer_encoder(x_temporal_encoded_player_time_flattened, mask=combined_input_mask)
        else:
            # next flatten all tokens into on sequence per batch. using temporal_mask to ensure only comparing tokens
            # from same players
            x_temporal_encoded_player_time_flattened = rearrange(x_temporal_encoded_batch_player_flattened,
                                                                 "(b p) t d -> b (p t) d", p=self.num_players,
                                                                 t=self.num_input_time_steps)
            combined_input_mask = combine_padding_sequence_masks(self.input_per_player_mask_cpu.to(x.device.type),
                                                                 dead_gathered_input, self.num_heads)
            x_temporal_embedded_player_time_flattened = \
                self.temporal_transformer_encoder(x_temporal_encoded_player_time_flattened, mask=combined_input_mask)
        # take last token per player
        x_temporal_embedded = rearrange(x_temporal_embedded_player_time_flattened, "b (p t) d -> b p t d",
                                        p=self.num_players, t=1 if self.drop_history else self.num_input_time_steps)
        if self.num_output_time_steps <= self.num_input_time_steps and not self.drop_history:
            x_temporal_embedded_flattened = rearrange(x_temporal_embedded[:, :, 0:self.num_output_time_steps, :],
                                                      "b p t d -> b (p t) d")
        else:
            x_temporal_embedded_repeated = repeat(x_temporal_embedded[:, :, [0], :],
                                                  "b p t d -> b p (t repeat) d", repeat=self.num_output_time_steps)
            x_temporal_embedded_flattened = rearrange(x_temporal_embedded_repeated, "b p t d -> b (p t) d")

        #x_pos_encoded_no_noise = self.encode_pos(x_pos, enable_noise=False)
        #if self.num_output_time_steps <= self.num_input_time_steps and not self.drop_history:
        #    x_gathered_no_noise = rearrange(
        #        torch.cat([x_pos_encoded_no_noise, x_crosshair, x_non_pos], -1)[:, :, 0:self.num_output_time_steps, :],
        #        "b p t d -> b (p t) d")
        #else:
        #    x_gathered_no_noise_repeated = repeat(torch.cat([x_pos_encoded_no_noise, x_crosshair, x_non_pos], -1)[:, :, [0], :],
        #                                          "b p t d -> b p (t repeat) d", repeat=self.num_output_time_steps)
        #    x_gathered_no_noise = rearrange(x_gathered_no_noise_repeated, "b p t d -> b (p t) d")
        #x_embedded_no_noise = self.embedding_model(x_gathered_no_noise)

        if self.player_mask_type == PlayerMaskType.EveryoneFullMask or \
                self.player_mask_type == PlayerMaskType.TeammateFullMask:
            combined_output_mask = combine_padding_sequence_masks(
                self.output_per_player_mask_cpu.to(x.device.type), dead_gathered_output, self.num_heads)
            transformed = self.spatial_transformer_encoder(x_temporal_embedded_flattened, mask=combined_output_mask)
        else:
            transformed = self.spatial_transformer_encoder(x_temporal_embedded_flattened,
                                                           src_key_padding_mask=dead_gathered_output)
        transformed_nested = rearrange(transformed, "b (p t) d -> b p t d",
                                       p=self.num_players, t=self.num_output_time_steps)
        latent = self.decoder(transformed_nested)
        prob_output = self.prob_output(latent / temperature)
        return latent, prob_output, one_hot_prob_to_index(prob_output)


def combine_padding_sequence_masks(sequence_mask: torch.Tensor, padding_mask: torch.Tensor, num_heads: int):
    # sequence mask: q,k - query is row, key is column, true if can't read from key to value
    # padding mask: b,k - batch is row, key is col, true if can't read from key for any query
    # need to replicate for each head
    sequence_for_each_batch_head_mask = repeat(sequence_mask, 'q k -> bn q k', bn=num_heads * padding_mask.shape[0])
    padding_in_sequence_format_mask = repeat(padding_mask, 'b k -> (b num_heads) q k', num_heads=num_heads,
                                             q=sequence_mask.shape[1])
    result = (sequence_for_each_batch_head_mask | padding_in_sequence_format_mask)
    # must disable mask on diagonal. Otherwise, can get NaN. If sequence_mask is only diagonal and padding_mask
    # removes a row, that row can't read from anywhere and becomes NaN
    for i in range(sequence_mask.shape[1]):
        result[:, i, i] = False
    return result