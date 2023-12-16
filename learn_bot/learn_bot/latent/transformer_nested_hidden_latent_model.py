from enum import Enum
from typing import List, Callable

from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.order.column_names import team_strs, player_team_str, all_prior_and_cur_ticks, \
    num_radial_ticks
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, num_radial_bins, \
    walking_modifier, ducking_modifier, default_similarity_columns
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import NavData, \
    one_hot_prob_to_index, max_speed_per_second
from learn_bot.latent.place_area.simulation.constants import weapon_scoped_to_max_speed_tensor
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR, CPU_DEVICE_STR, \
    range_list_to_index_list, flatten_list
from learn_bot.libs.positional_encoding import *
from einops import rearrange, repeat
from positional_encodings.torch_encodings import PositionalEncoding1D


def get_player_columns_by_str(cts: IOColumnTransformers, contained_str: str) -> List[int]:
    return flatten_list(
        [range_list_to_index_list(
            cts.get_name_ranges(True, False, contained_str=contained_str + " " + player_team_str(team_str, player_index)))
            for team_str in team_strs for player_index in range(max_enemies)]
    )


def get_columns_by_str(cts: IOColumnTransformers, contained_str: str) -> List[int]:
    return range_list_to_index_list(cts.get_name_ranges(True, False, contained_str=contained_str))


d2_min = [-2257., -1207., -204.128]
d2_max = [1832., 3157., 236.]
stature_to_speed_list = [1., walking_modifier, ducking_modifier]


class PlayerMaskType(Enum):
    NoMask = 1
    EveryoneMask = 3

    def __str__(self) -> str:
        if self == PlayerMaskType.NoMask:
            return "NoMask"
        else:
            return "EveryoneMask"


class OutputMaskType(Enum):
    NoMask = 1
    EngagementMask = 2
    NoEngagementMask = 3

    def __str__(self) -> str:
        if self == OutputMaskType.NoMask:
            return "NoMask"
        elif self == OutputMaskType.EngagementMask:
            return "EngagementMask"
        else:
            return "NoEngagementMask"


class ControlType(Enum):
    NoControl = 1
    TimeControl = 2
    SimilarityControl = 3

    def __str__(self) -> str:
        if self == ControlType.NoControl:
            return "NoControl"
        elif self == ControlType.TimeControl:
            return "TimeControl"
        else:
            return "SimilarityControl"


class TransformerNestedHiddenLatentModel(nn.Module):
    internal_width: int
    cts: IOColumnTransformers
    noise_var: float

    def __init__(self, cts: IOColumnTransformers, internal_width: int, num_players: int, num_input_time_steps: int,
                 num_layers: int, num_heads: int, control_type: ControlType, player_mask_type: PlayerMaskType,
                 mask_partial_info: bool):
        super(TransformerNestedHiddenLatentModel, self).__init__()
        self.cts = cts
        self.internal_width = internal_width
        self.mask_partial_info = mask_partial_info
        # transformed/transformed doesn't matter since no angles and all categorical variables are
        # "distirbution" type meaning pre one hot encoded
        all_c4_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="c4"))
        #_, c4_columns_names = cts.get_name_ranges(True, True, contained_str="c4", include_names=True)
        c4_decrease_distance_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="decrease distance to c4"))
        non_player_c4_columns_ranges = [i for i in all_c4_columns_ranges if i not in c4_decrease_distance_columns_ranges]
        #baiting_columns_ranges = range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="baiting"))
        players_columns = [non_player_c4_columns_ranges +  #baiting_columns_ranges +
                           range_list_to_index_list(cts.get_name_ranges(True, True, contained_str=" " + player_team_str(team_str, player_index)))
                           for team_str in team_strs for player_index in range(max_enemies)]
        #players_columns_names = [cts.get_name_ranges(True, True, contained_str=" " + player_team_str(team_str, player_index), include_names=True)[1]
        #                         for team_str in team_strs for player_index in range(max_enemies)]
        self.num_players = len(players_columns)
        self.num_dim = 3
        self.num_heads = num_heads

        all_players_pos_columns = get_player_columns_by_str(cts, "player pos")
        all_players_pos_columns_tensor = torch.IntTensor(all_players_pos_columns)
        self.nested_players_pos_columns_tensor = rearrange(all_players_pos_columns_tensor, '(p t d) -> p t d',
                                                      p=self.num_players, t=all_prior_and_cur_ticks, d=self.num_dim)
        self.players_pos_columns = rearrange(self.nested_players_pos_columns_tensor[:, 0:num_input_time_steps, :],
                                             'p t d -> (p t d)',
                                             p=self.num_players, t=num_input_time_steps, d=self.num_dim).tolist()
        all_players_nearest_crosshair_to_enemy_columns = \
            get_player_columns_by_str(cts, "player nearest crosshair distance to enemy")
        all_players_nearest_crosshair_to_enemy_columns_tensor = \
                    torch.IntTensor(all_players_nearest_crosshair_to_enemy_columns)
        nested_players_nearest_crosshair_to_enemy_columns_tensor = \
            rearrange(all_players_nearest_crosshair_to_enemy_columns_tensor, '(p t) -> p t',
                      p=self.num_players, t=all_prior_and_cur_ticks)
        self.players_nearest_crosshair_to_enemy_columns = \
            rearrange(nested_players_nearest_crosshair_to_enemy_columns_tensor[:, 0:num_input_time_steps],
                      'p t -> (p t)', p=self.num_players, t=num_input_time_steps).tolist()
        self.players_visibility = \
            range_list_to_index_list(cts.get_name_ranges(True, True, contained_str="player enemy visible"))

        self.players_all_temporal_columns = all_players_pos_columns + all_players_nearest_crosshair_to_enemy_columns
        self.players_non_temporal_columns = flatten_list([
            [player_column for player_column in player_columns
             if player_column not in self.players_all_temporal_columns]
            for player_columns in players_columns
        ])
        self.num_similarity_columns = default_similarity_columns
        self.num_input_time_steps = num_input_time_steps
        self.num_output_time_steps = num_radial_ticks
        assert self.num_input_time_steps == 1 or self.num_input_time_steps == self.num_output_time_steps
        self.control_type = control_type
        self.time_control_columns = get_columns_by_str(cts, "player decrease distance to c4")

        # ensure player counts right
        assert self.num_players == num_players
        self.num_players_per_team = self.num_players // 2

        # compute mask to isolate everyone
        self.input_per_player_mask_cpu = torch.zeros([self.num_players, self.num_players], dtype=torch.bool)
        if player_mask_type == PlayerMaskType.EveryoneMask:
            for i in range(self.num_players):
                for j in range(self.num_players):
                    self.input_per_player_mask_cpu[i, j] = i != j

        self.player_mask_type = player_mask_type

        # get alive columns
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
        self.weapon_scoped_to_max_speed_tensor_gpu = weapon_scoped_to_max_speed_tensor.to(CUDA_DEVICE_STR)


        # NERF code calls it positional embedder, but it's encoder since not learned
        self.spatial_positional_encoder, self.spatial_positional_encoder_out_dim = get_embedder()
        self.columns_per_player = (len(self.players_non_temporal_columns) // self.num_players) + \
                                  self.num_similarity_columns + \
                                  self.spatial_positional_encoder_out_dim * num_input_time_steps + \
                                  num_input_time_steps # for the nearest_crosshair_to_enemy_columns

        # positional encoding library doesn't play well with torchscript, so I'll just make the
        # encoding matrices upfront and add them during inference
        temporal_positional_encoder = PositionalEncoding1D(self.internal_width)
        self.temporal_positional_encoding = \
            temporal_positional_encoder(torch.zeros([self.num_players, self.num_output_time_steps, self.internal_width]))

        # build actual model layers
        self.embedding_model = nn.Sequential(
            nn.Linear(self.columns_per_player, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
            nn.LeakyReLU(),
            nn.Linear(self.internal_width, self.internal_width),
        )

        self.nav_data_cpu = NavData(CPU_DEVICE_STR)
        self.nav_data_cuda = NavData(CUDA_DEVICE_STR)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.internal_width, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.decoder = nn.Sequential(
            nn.Linear(self.internal_width, num_radial_bins),
        )

        self.prob_output = nn.Sequential(
            nn.Softmax(dim=3),
        )

    def encode_pos(self, pos: torch.tensor, enable_noise=True):
        # https://arxiv.org/pdf/2003.08934.pdf
        # 5.1 - everything is normalized -1 to 1
        pos_scaled = torch.zeros_like(pos)
        if pos.device.type == CUDA_DEVICE_STR:
            pos_scaled = (pos - self.d2_min_gpu) / (self.d2_max_gpu - self.d2_min_gpu)
        else:
            pos_scaled = (pos - self.d2_min_cpu) / (self.d2_max_cpu - self.d2_min_cpu)
        #if self.num_input_time_steps > 1:
        #    # + rather than - here as speed is positive, d2 dimensions are negative
        #    # multiply by 2 so be sure to capture outliers
        #    pos_scaled[:, :, 1:] = (pos[:, :, 1:] + (2 * max_run_speed_per_sim_tick)) / (4 * max_run_speed_per_sim_tick)
        pos_scaled = torch.clamp(pos_scaled, 0, 1)
        pos_scaled = (pos_scaled * 2) - 1

        if self.noise_var >= 0. and enable_noise:
            means = torch.zeros(pos.shape)
            vars = torch.full(pos.shape, self.noise_var)
            noise = torch.normal(means, vars).to(pos.device.type)
            noise_scaled = noise / (self.d2_max_gpu - self.d2_min_gpu)
            pos_scaled += noise_scaled

        return self.spatial_positional_encoder(pos_scaled)

    def forward(self, x_in, similarity_in, temperature):
        x = x_in.clone()
        if self.control_type != ControlType.TimeControl:
            x[:, self.time_control_columns] = 0.
        if self.mask_partial_info:
            x[:, self.players_visibility] = 0.
        # legacy, so that if hand in too many similarity columns, still use them
        similarity = similarity_in[:, self.num_similarity_columns * -1:].clone()
        if self.control_type != ControlType.SimilarityControl:
            similarity[:, :] = 0.
        x_pos = rearrange(x[:, self.players_pos_columns], "b (p t d) -> b p t d", p=self.num_players,
                          t=self.num_input_time_steps, d=self.num_dim)
        x_pos_encoded = self.encode_pos(x_pos)
        x_crosshair = rearrange(x[:, self.players_nearest_crosshair_to_enemy_columns], "b (p t) -> b p t 1",
                                p=self.num_players, t=self.num_input_time_steps)

        x_non_pos_without_similarity = \
            rearrange(x[:, self.players_non_temporal_columns], "b (p t d) -> b p t d",
                      p=self.num_players, t=1)
        similarity_expanded = rearrange(similarity, '(b p t) d -> b p t d', p=1, t=1) \
            .repeat([1, self.num_players, 1, 1])
        x_non_pos = torch.concat([x_non_pos_without_similarity, similarity_expanded], dim=-1)

        # extend in temporal (if necessary) and embed individual tokens
        x_non_pos_temporal = repeat(x_non_pos, 'b p 1 d -> b p repeat d', repeat=self.num_output_time_steps)
        if self.num_input_time_steps == self.num_output_time_steps:
            x_pos_temporal = x_pos_encoded
            x_crosshair_temporal = x_crosshair
        else:
            x_pos_temporal = repeat(x_pos_encoded, 'b p 1 d -> b p repeat d',
                                    repeat=self.num_output_time_steps)
            x_crosshair_temporal = repeat(x_crosshair, 'b p 1 d -> b p repeat d',
                                    repeat=self.num_output_time_steps)
        if self.mask_non_pos:
            x_gathered = torch.cat([x_pos_temporal, x_crosshair_temporal,
                                    torch.zeros_like(x_non_pos_temporal)], -1)
        else:
            x_gathered = torch.cat([x_pos_temporal, x_crosshair_temporal, x_non_pos_temporal], -1)
        #x_gathered = torch.cat([x_pos_encoded, x_vel_scaled, x_non_pos], -1)
        x_embedded = self.embedding_model(x_gathered)

        # add temporal enocding to separate tokens
        x_batch_player_flattened = rearrange(x_embedded, "b p t d -> (b p) t d")
        batch_temporal_positional_encoding = self.temporal_positional_encoding.repeat([x.shape[0], 1, 1]) \
            .to(x.device.type)
        x_batch_player_flattened_temporal_with_encoding = x_batch_player_flattened + batch_temporal_positional_encoding
        x_player_time_flattened_temporal_with_encoding = rearrange(x_batch_player_flattened_temporal_with_encoding,
                                                                   '(b p) t d -> b (p t) d',
                                                                   p=self.num_players, t=self.num_output_time_steps)


        # figure out alive and dead for masking out dead ploayers
        alive_gathered = x[:, self.alive_columns]
        dead_gathered = alive_gathered < 0.5

        combined_mask = combine_padding_sequence_masks(self.input_per_player_mask_cpu.to(x.device.type),
                                                             dead_gathered, self.num_heads)
        combined_temporal_mask = repeat(combined_mask, 'b p0 p1 -> b (p0 repeat0) (p1 repeat1)',
                                        repeat0=self.num_output_time_steps,
                                        repeat1=self.num_output_time_steps)

        transformer_output = self.transformer_encoder(x_player_time_flattened_temporal_with_encoding,
                                                      mask=combined_temporal_mask)
        # keep output temporal nesting for now, rest of system assumes it exists
        transformed_nested = rearrange(transformer_output, "b (p t) d -> b p t d",
                                       p=self.num_players, t=self.num_output_time_steps)
        latent = self.decoder(transformed_nested)
        prob_output = self.prob_output(latent / temperature)
        # output 0 is batch, 1 is players, 2 is output time step, 3 is probabilities/logits
        return latent, prob_output, one_hot_prob_to_index(prob_output)


def combine_padding_sequence_masks(sequence_mask: torch.Tensor, padding_mask: torch.Tensor, num_heads: int,
                                   enable_diagonal: bool = True):
    # sequence mask: q,k - query is row, key is column, true if can't read from key to value
    # padding mask: b,k - batch is row, key is col, true if can't read from key for any query
    # need to replicate for each head
    sequence_for_each_batch_head_mask = repeat(sequence_mask, 'q k -> bn q k', bn=num_heads * padding_mask.shape[0])
    padding_in_sequence_format_mask = repeat(padding_mask, 'b k -> (b num_heads) q k', num_heads=num_heads,
                                             q=sequence_mask.shape[1])
    result = (sequence_for_each_batch_head_mask | padding_in_sequence_format_mask)
    # must disable mask on diagonal. Otherwise, can get NaN. If sequence_mask is only diagonal and padding_mask
    # removes a row, that row can't read from anywhere and becomes NaN
    if enable_diagonal:
        for i in range(sequence_mask.shape[1]):
            result[:, i, i] = False
    return result