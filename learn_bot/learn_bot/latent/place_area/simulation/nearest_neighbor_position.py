from typing import List

import torch

from learn_bot.latent.analyze.knn.knn_position import get_nearest_neighbors_one_situation
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.simulation.constants import num_time_steps
from learn_bot.latent.place_area.simulation.simulator import RoundLengths
from learn_bot.libs.io_transforms import flatten_list
from learn_bot.libs.vec import Vec3


def update_nn_position_rollout_tensor(loaded_model: LoadedModel, round_lengths: RoundLengths,
                                      ground_truth_rollout_tensor: torch.Tensor) -> torch.Tensor:
    nn_rollout_tensor = ground_truth_rollout_tensor.clone().detach()

    # the tick indices to check similarity on, rest will be copied from these positions
    similarity_tick_indices = flatten_list([
        [idx for idx in round_subset_tick_indices if (idx - round_subset_tick_indices.start) % num_time_steps == 0]
        for _, round_subset_tick_indices in round_lengths.round_to_subset_tick_indices.items()
    ])

    points_for_nn_tensor = ground_truth_rollout_tensor[similarity_tick_indices]

    for point_index in range(points_for_nn_tensor.shape[0]):
        ct_pos: List[Vec3] = []
        t_pos: List[Vec3] = []

        # get the pos for all the alive players in the point
        for player_index, alive_column_index in enumerate(loaded_model.model.alive_columns):
            pos_columns = loaded_model.model.nested_players_pos_columns_tensor[player_index, 0].tolist()
            if points_for_nn_tensor[point_index, alive_column_index] == 1:
                pos = Vec3(
                    points_for_nn_tensor[point_index, pos_columns[0]],
                    points_for_nn_tensor[point_index, pos_columns[1]],
                    points_for_nn_tensor[point_index, pos_columns[2]]
                )
                if specific_player_place_area_columns[player_index].is_ct:
                    ct_pos.append(pos)
                else:
                    t_pos.append(pos)

        # 2 as 1 should be this, second should be second best
        same_and_nearest_nps = get_nearest_neighbors_one_situation(ct_pos, t_pos, 2, loaded_model, '', 0, False, False,
                                                                   num_time_steps)
        nn_rollout_tensor[point_index:point_index+num_time_steps] = torch.tensor(same_and_nearest_nps[1])

    return nn_rollout_tensor






