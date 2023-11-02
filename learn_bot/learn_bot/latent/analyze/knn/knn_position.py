from typing import List

import numpy as np
import pandas as pd
import math

from learn_bot.latent.analyze.knn.generate_player_index_mappings import generate_all_player_to_column_mappings
from learn_bot.latent.analyze.knn.select_alive_players import get_id_df_and_alive_pos_np
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.train import load_data_options
from learn_bot.libs.vec import Vec3

l2_distance_col = 'l2 distance'
hdf5_id_col = 'hdf5 id'


def get_nearest_neighbors(num_ct_alive: int, ct_pos: List[Vec3], num_t_alive: int, t_pos: List[Vec3],
                          num_matches: int = 100) -> pd.DataFrame:
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result)

    players_pos = ct_pos + t_pos

    player_to_column_mappings = generate_all_player_to_column_mappings(num_ct_alive, num_t_alive)

    min_distance_dfs: List[pd.DataFrame] = []

    print("computing rounds with minimum distance points")
    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        print(f"Processing hdf5 {i} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
        id_df, alive_pos_np = get_id_df_and_alive_pos_np(hdf5_wrapper, loaded_model.model, num_ct_alive, num_t_alive)

        base_point_np = np.zeros_like(alive_pos_np)
        for i, player_pos in enumerate(players_pos):
            base_point_np[:, i, 0] = player_pos.x
            base_point_np[:, i, 1] = player_pos.y
            base_point_np[:, i, 2] = player_pos.z


        # find min distance from each point to the base point across all player mappings
        min_distance_per_row = np.zeros(alive_pos_np.shape[0])
        min_distance_per_row[:] = math.inf
        for player_to_column_mapping in player_to_column_mappings:
            mapping_distance = np.zeros(alive_pos_np.shape[0])
            # distance per player for this mapping
            for player_index, column_index in player_to_column_mapping.player_to_column:
                player_distance = (
                        (alive_pos_np[:, column_index, 0] - base_point_np[:, player_index].x) ** 2. +
                        (alive_pos_np[:, column_index, 1] - base_point_np[:, player_index].y) ** 2. +
                        (alive_pos_np[:, column_index, 2] - base_point_np[:, player_index].z) ** 2.
                ) ** .5
                mapping_distance += np.sum(player_distance, axis=1)
            min_distance_per_row = \
                np.where(mapping_distance < min_distance_per_row, mapping_distance, min_distance_per_row)

        id_with_distance_df = id_df.copy()
        id_with_distance_df[l2_distance_col] = min_distance_per_row
        id_with_distance_df[hdf5_id_col] = i

        min_distance_per_round_df = id_with_distance_df.groupby(round_id_column, as_index=False).min(l2_distance_col) \
            .sort_values(l2_distance_col).iloc[:num_matches]
        min_distance_dfs.append(min_distance_per_round_df)

    min_distance_df = pd.DataFrame(min_distance_dfs).sort_values(l2_distance_col).iloc[:num_matches]



