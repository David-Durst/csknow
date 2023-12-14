import time
from typing import List, Optional

import torch
from einops import rearrange
from tqdm import tqdm

from learn_bot.latent.analyze.knn.knn_by_position import get_nearest_neighbors_one_situation, CachedNNData
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
    similarity_rollout_indices = [round_lengths.round_id_to_list_id[round_id] * round_lengths.max_length_per_round + tick_index
                                  for round_id, round_tick_range in round_lengths.round_to_tick_ids.items()
                                  for tick_index in range(len(round_tick_range))
                                  if tick_index % num_time_steps == 0]

    similarity_tick_indices = [tick_index
                               for round_id, round_subset_tick_range in round_lengths.round_to_subset_tick_indices.items()
                               for i, tick_index in enumerate(round_subset_tick_range)
                               if i % num_time_steps == 0]

    similarity_round_ids = [round_id
                            for round_id, round_tick_range in round_lengths.round_to_tick_ids.items()
                            for tick_index in range(len(round_tick_range))
                            if tick_index % num_time_steps == 0]


    points_for_nn_tensor = ground_truth_rollout_tensor[similarity_rollout_indices]
    cached_nn_data = CachedNNData()
    prior_ct_alive = 0
    prior_t_alive = 0

    start_time = time.time()
    with tqdm(total=points_for_nn_tensor.shape[0], disable=False) as pbar:
        for point_index in range(points_for_nn_tensor.shape[0]):
            #print(point_index)
            #if point_index in range(8, 12):
            #    tick_index = similarity_tick_indices[point_index]
            #    print(loaded_model.get_cur_id_df().iloc[tick_index])
            #if point_index == 11:
            #    print("hi")
            #if point_index != 12:
            #    continue

            alive_player_indices: List[int] = []
            dead_player_indices: List[int] = []
            ct_pos: List[Vec3] = []
            t_pos: List[Vec3] = []

            # get the pos for all the alive players in the point
            x_pos_columns = []
            for player_index, alive_column_index in enumerate(loaded_model.model.alive_columns):
                pos_columns = loaded_model.model.nested_players_pos_columns_tensor[player_index, 0].tolist()
                x_pos_columns.append(pos_columns[0])
                if points_for_nn_tensor[point_index, alive_column_index] == 1:
                    alive_player_indices.append(player_index)
                    pos = Vec3(
                        points_for_nn_tensor[point_index, pos_columns[0]],
                        points_for_nn_tensor[point_index, pos_columns[1]],
                        points_for_nn_tensor[point_index, pos_columns[2]]
                    )
                    if specific_player_place_area_columns[player_index].is_ct:
                        ct_pos.append(pos)
                    else:
                        t_pos.append(pos)
                else:
                    dead_player_indices.append(player_index)

            if prior_ct_alive != len(ct_pos) or prior_t_alive != len(t_pos):
                cached_nn_data = CachedNNData()
            #print(f"point index {point_index}, round id {similarity_round_ids[point_index]}, num ct {len(ct_pos)}, num t {len(t_pos)}")
            # 2 as 1 should be this, second should be second best
            same_and_nearest_nps, player_to_full_table_id, l2_distances, cached_nn_data = \
                get_nearest_neighbors_one_situation(ct_pos, t_pos, 2, loaded_model, '', 0, False, False,
                                                    num_time_steps, cached_nn_data, limit_matches_by_time_left=False)

            # usually index 1 as skipping first match, first match is first match is input pos
            # but if end of round, not 5 seconds to match, then may not include it and may need first entry as that's best match and not the same
            if l2_distances.iloc[0] == 0.:
                #print('found nearest')
                nearest_np = same_and_nearest_nps[1].copy()
                selected_player_to_full_table_id = player_to_full_table_id[1]
            else:
                print('didnt find nearest')
                nearest_np = same_and_nearest_nps[0].copy()
                selected_player_to_full_table_id = player_to_full_table_id[0]
            orig_order_nearest_np = nearest_np.copy()

            # sort columns and then insert
            # make sure to get dead players columns too, nearest neighbor matching only worries abouve alive
            # player means alive players used in mapping to full table, unused entries in full table are dead players
            # not included in mapping (since dead)
            unused_full_table_id = [i for i in range(len(loaded_model.model.alive_columns)) if i not in selected_player_to_full_table_id]
            player_and_unused_full_table_id = selected_player_to_full_table_id.tolist() + unused_full_table_id
            alive_and_dead_pos_columns = rearrange(loaded_model.model.nested_players_pos_columns_tensor[alive_player_indices + dead_player_indices, 0], 'p d -> (p d)').tolist()
            alive_and_dead_alive_columns = [loaded_model.model.alive_columns[i] for i in alive_player_indices + dead_player_indices]
            player_and_unused_to_full_table_pos_columns = rearrange(loaded_model.model.nested_players_pos_columns_tensor[player_and_unused_full_table_id, 0], 'p d -> (p d)').tolist()
            player_and_unused_to_full_table_alive_columns = [loaded_model.model.alive_columns[i] for i in player_and_unused_full_table_id]
            #unmodified_point = points_for_nn_tensor[point_index].clone()

            nearest_np[:, alive_and_dead_pos_columns] = nearest_np[:, player_and_unused_to_full_table_pos_columns]
            nearest_np[:, alive_and_dead_alive_columns] = nearest_np[:, player_and_unused_to_full_table_alive_columns]

            nearest_tensor = torch.tensor(nearest_np, dtype=nn_rollout_tensor.dtype)
            # if not a full length in round
            if nearest_tensor.shape[0] < num_time_steps:
                full_nearest_tensor = torch.zeros([num_time_steps, nearest_tensor.shape[1]], dtype=nearest_tensor.dtype)
                full_nearest_tensor[:nearest_tensor.shape[0]] = nearest_tensor
                nearest_tensor = full_nearest_tensor

            nn_rollout_tensor[similarity_rollout_indices[point_index]:similarity_rollout_indices[point_index] + num_time_steps, alive_and_dead_pos_columns] = \
                nearest_tensor[:, alive_and_dead_pos_columns]

            #check_for_alive_with_dead_positions(nn_rollout_tensor[similarity_rollout_indices[point_index]:
            #                                                      similarity_rollout_indices[point_index] + num_time_steps],
            #                                    nn_rollout_tensor[similarity_rollout_indices[point_index]:
            #                                                      similarity_rollout_indices[point_index] + num_time_steps],
            #                                    loaded_model)
            fill_dead_positions_with_last_alive(nn_rollout_tensor[similarity_rollout_indices[point_index]:
                                                                  similarity_rollout_indices[point_index] + num_time_steps],
                                                nearest_tensor,
                                                loaded_model)
            check_for_alive_with_dead_positions(nn_rollout_tensor[similarity_rollout_indices[point_index]:
                                                                  similarity_rollout_indices[point_index] + num_time_steps],
                                                nearest_tensor,
                                                loaded_model)

            prior_ct_alive = len(ct_pos)
            prior_t_alive = len(t_pos)
            pbar.update(1)
            #if point_index == 99:
            #    total_time = time.time() - start_time
            #    print(f"nearest neighbor position time {total_time}, time per point {total_time / (point_index + 1)}")
        #    quit(0)

    return nn_rollout_tensor

num_dead_while_orig_alive = 0
def check_for_alive_with_dead_positions(orig_rollout_tensor: torch.Tensor, updated_rollout_tensor: torch.Tensor, loaded_model: LoadedModel):
    global num_dead_while_orig_alive
    for player_index, alive_column_index in enumerate(loaded_model.model.alive_columns):
        pos_columns = loaded_model.model.nested_players_pos_columns_tensor[player_index, 0].tolist()
        alive_column = orig_rollout_tensor[:, alive_column_index]
        updated_alive_column = updated_rollout_tensor[:, alive_column_index]
        dead_pos_alive = (alive_column > 0) & (orig_rollout_tensor[:, pos_columns[0]] == 0.) & (orig_rollout_tensor[:, pos_columns[1]] == 0.)
        if sum(dead_pos_alive) > 0:
            num_dead_while_orig_alive = sum(dead_pos_alive)
            print(f"num dead while orig alive {num_dead_while_orig_alive}")

def fill_dead_positions_with_last_alive(orig_rollout_tensor: torch.Tensor, updated_rollout_tensor: torch.Tensor, loaded_model: LoadedModel):
    for player_index, alive_column_index in enumerate(loaded_model.model.alive_columns):
        pos_columns = loaded_model.model.nested_players_pos_columns_tensor[player_index, 0].tolist()
        # these are orig alive ticks, not nearest neighrbor alive ticks, so will miss 0's
        num_orig_alive_ticks = int(torch.sum(orig_rollout_tensor[:, alive_column_index]))
        num_updated_alive_ticks = int(torch.sum(updated_rollout_tensor[:, alive_column_index]))
        if num_orig_alive_ticks > num_updated_alive_ticks:
            orig_rollout_tensor[num_updated_alive_ticks:num_orig_alive_ticks, pos_columns] = orig_rollout_tensor[num_updated_alive_ticks-1, pos_columns]
        #if num_alive_ticks < rollout_tensor.shape[0]:
        #    rollout_tensor[num_alive_ticks:, pos_columns] = rollout_tensor[max(num_alive_ticks-1, 0), pos_columns]



