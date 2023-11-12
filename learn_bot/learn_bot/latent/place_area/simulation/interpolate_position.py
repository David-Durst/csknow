from dataclasses import dataclass
from typing import Tuple, Dict

import torch
from einops import repeat

from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.simulation.constants import num_time_steps
from learn_bot.latent.place_area.simulation.simulator import RoundLengths


@dataclass
class RoundStartEndLength:
    round_cur_index: int
    interpolation_start_index: int
    interpolation_end_index: int
    player_max_alive_steps_in_interpolation: torch.Tensor

    def get_percent_end(self) -> torch.Tensor:
        cur_steps = self.round_cur_index - self.interpolation_start_index
        cur_alive_steps = torch.where(self.player_max_alive_steps_in_interpolation < cur_steps,
                                      self.player_max_alive_steps_in_interpolation, cur_steps)
        max_steps = self.interpolation_end_index - self.interpolation_start_index
        max_alive_steps = torch.where(self.player_max_alive_steps_in_interpolation < max_steps,
                                      self.player_max_alive_steps_in_interpolation, max_steps)
        max_alive_steps = torch.where(max_alive_steps < 1., 1., max_alive_steps)
        return cur_alive_steps / max_alive_steps

    def get_percent_start(self) -> torch.Tensor:
        return 1 - self.get_percent_end()


def fix_dead_positions(loaded_model: LoadedModel, round_lengths: RoundLengths,
                       ground_truth_rollout_tensor: torch.tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    ground_truth_rollout_tensor = ground_truth_rollout_tensor.clone().detach()
    round_id_to_player_max_alive_steps: Dict[int, torch.Tensor] = {}

    for round_index, round_id in enumerate(round_lengths.round_ids):
        round_start_index = round_index * round_lengths.max_length_per_round
        round_end_index = round_start_index + round_lengths.round_to_length[round_id] - 1
        round_rollout_tensor = ground_truth_rollout_tensor[round_start_index:(round_end_index + 1)]
        round_id_to_player_max_alive_steps[round_id] = torch.zeros([len(specific_player_place_area_columns)])

        for column_index in range(len(specific_player_place_area_columns)):
            player_pos_column_indices = loaded_model.model.nested_players_pos_columns_tensor[column_index, 0].tolist()
            player_alive_column_index = loaded_model.model.alive_columns[column_index]
            player_alive = repeat(round_rollout_tensor[:, player_alive_column_index] == 1, 't -> t d',
                                  d=len(player_pos_column_indices))
            player_max_alive_steps = max(0, player_alive[:, 0].sum() - 1)
            round_id_to_player_max_alive_steps[round_id][column_index] = player_max_alive_steps
            #if player_max_alive_steps > 0 and player_max_alive_steps < len(round_rollout_tensor) - 1:
            #    print('hi')
            player_max_alive_pos = round_rollout_tensor[player_max_alive_steps, player_pos_column_indices]

            round_rollout_tensor[:, player_pos_column_indices] = \
                torch.where(player_alive,
                            round_rollout_tensor[:, player_pos_column_indices],
                            player_max_alive_pos)

    return ground_truth_rollout_tensor, round_id_to_player_max_alive_steps


def update_interpolation_position_rollout_tensor(loaded_model: LoadedModel, round_lengths: RoundLengths,
                                                 ground_truth_rollout_tensor: torch.Tensor,
                                                 interpolate_to_end_of_round: bool) -> torch.Tensor:
    ground_truth_rollout_tensor, round_id_to_player_max_alive_steps = \
        fix_dead_positions(loaded_model, round_lengths, ground_truth_rollout_tensor)
    round_start_end_lengths = []
    for round_index, round_id in enumerate(round_lengths.round_ids):
        for step_index in range(round_lengths.round_to_length[round_id]):
            round_start_index = round_index * round_lengths.max_length_per_round
            round_end_index = round_start_index + round_lengths.round_to_length[round_id] - 1
            if interpolate_to_end_of_round:
                interpolation_start_index = round_start_index
                interpolation_end_index = round_end_index
            else:
                interpolation_start_index = (step_index // num_time_steps * num_time_steps) + round_start_index
                interpolation_end_index = min(
                    ((step_index // num_time_steps + 1) * num_time_steps) - 1 + round_start_index,
                    round_end_index
                )

            interpolation_start_step_index = interpolation_start_index - round_start_index
            player_max_alive_steps_in_interpolation = round_id_to_player_max_alive_steps[round_id] - \
                                                      interpolation_start_step_index
            player_max_alive_steps_in_interpolation = torch.where(
                player_max_alive_steps_in_interpolation > 0.,
                player_max_alive_steps_in_interpolation,
                0.
            )

            #if round_index == 1 and step_index == 108:
            #    print('hi')

            round_start_end_lengths.append(RoundStartEndLength(
                round_start_index + step_index, interpolation_start_index, interpolation_end_index,
                player_max_alive_steps_in_interpolation
            ))

    pos_column_indices = []
    for column_index in range(len(specific_player_place_area_columns)):
        pos_column_indices += loaded_model.model.nested_players_pos_columns_tensor[column_index, 0].tolist()

    interpolation_rollout_tensor = ground_truth_rollout_tensor.clone().detach()
    for round_start_end_length in round_start_end_lengths:
        start_pos = ground_truth_rollout_tensor[round_start_end_length.interpolation_start_index, pos_column_indices]
        end_pos = ground_truth_rollout_tensor[round_start_end_length.interpolation_end_index, pos_column_indices]
        # need to repeat once per pos dimension
        percent_start_repeated = repeat(round_start_end_length.get_percent_start(), 'b -> (b d)', d=3)
        percent_end_repeated = repeat(round_start_end_length.get_percent_end(), 'b -> (b d)', d=3)
        #if round_start_end_length.round_cur_index == 437:
        #    print('hi')
        #    percent_end_repeated = repeat(round_start_end_length.get_percent_end(), 'b -> (b d)', d=3)
        interpolation_rollout_tensor[round_start_end_length.round_cur_index, pos_column_indices] = \
            start_pos * percent_start_repeated + end_pos * percent_end_repeated

    return interpolation_rollout_tensor
