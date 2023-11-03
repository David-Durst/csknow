from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import math

from learn_bot.latent.analyze.knn.generate_player_index_mappings import generate_all_player_to_column_mappings
from learn_bot.latent.analyze.knn.plot_min_distance_rounds import plot_min_distance_rounds, l2_distance_col, hdf5_id_col
from learn_bot.latent.analyze.knn.select_alive_players import get_id_df_and_alive_pos_np
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.load_model import load_model_file, LoadedModel
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.train import load_data_options
from learn_bot.libs.vec import Vec3


@dataclass
class PositionSituationParameters:
    ct_pos: List[Vec3]
    t_pos: List[Vec3]
    name: str


def get_nearest_neighbors(situations: List[PositionSituationParameters], num_matches: int = 100) -> pd.DataFrame:
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, load_pd=False)
    load_data_result.multi_hdf5_wrapper.create_np_arrays(loaded_model.model.cts)

    for situation in situations:
        print(f"processing {situation.name}")
        get_nearest_neighbors_one_situation(situation.ct_pos, situation.t_pos, num_matches,
                                            loaded_model, situation.name)


def get_nearest_neighbors_one_situation(ct_pos: List[Vec3], t_pos: List[Vec3], num_matches: int,
                                        loaded_model: LoadedModel, situation_name: str):
    num_ct_alive = len(ct_pos)
    num_t_alive = len(t_pos)

    players_pos = ct_pos + t_pos

    player_to_column_mappings = generate_all_player_to_column_mappings(num_ct_alive, num_t_alive)

    min_distance_rounds_per_hdf5: List[pd.DataFrame] = []

    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        id_df, alive_pos_np = get_id_df_and_alive_pos_np(hdf5_wrapper, loaded_model.model, num_ct_alive, num_t_alive)
        alive_pos_np = alive_pos_np.astype(np.float32)

        base_point_np = np.zeros_like(alive_pos_np)
        for j, player_pos in enumerate(players_pos):
            base_point_np[:, j, 0] = player_pos.x
            base_point_np[:, j, 1] = player_pos.y
            base_point_np[:, j, 2] = player_pos.z


        # find min distance from each point to the base point across all player mappings
        min_distance_per_row = np.zeros(alive_pos_np.shape[0])
        min_distance_per_row[:] = math.inf
        for player_to_column_mapping in player_to_column_mappings:
            # sum distance per player for this mapping
            player_indices = []
            column_indices = []
            for player_index, column_index in player_to_column_mapping.player_to_column.items():
                player_indices.append(player_index)
                column_indices.append(column_index)
            mapping_distance_per_player = (
                    (alive_pos_np[:, column_indices, 0] - base_point_np[:, player_indices, 0]) ** 2. +
                    (alive_pos_np[:, column_indices, 1] - base_point_np[:, player_indices, 1]) ** 2. +
                    (alive_pos_np[:, column_indices, 2] - base_point_np[:, player_indices, 2]) ** 2.
            ) ** .5
            mapping_distance = np.sum(mapping_distance_per_player, axis=1)
            min_distance_per_row = \
                np.where(mapping_distance < min_distance_per_row, mapping_distance, min_distance_per_row)

        id_with_distance_df = id_df.copy()
        id_with_distance_df[l2_distance_col] = min_distance_per_row
        id_with_distance_df[hdf5_id_col] = i

        min_distance_per_round_df = id_with_distance_df.groupby(round_id_column, as_index=False).min(l2_distance_col) \
            .sort_values(l2_distance_col).iloc[:num_matches]
        min_distance_rounds_per_hdf5.append(min_distance_per_round_df)

    min_distance_rounds_df = pd.concat(min_distance_rounds_per_hdf5).sort_values(l2_distance_col).iloc[:num_matches]
    plot_min_distance_rounds(loaded_model, min_distance_rounds_df, situation_name)


defend_a_cat_parameters = PositionSituationParameters(
    [Vec3(563.968750, 2763.999511, 97.379516), Vec3(357.684234, 1650.239990, 27.671302)],
    [Vec3(1160.000976, 2573.304931, 96.338958)],
    "DefendACat"
)
defend_a_cat_teammates_behind_parameters = PositionSituationParameters(
    [Vec3(563.968750, 2763.999511, 97.379516), Vec3(357.684234, 1650.239990, 27.671302)],
    [Vec3(1160.000976, 2573.304931, 96.338958), Vec3(1175.846923, 2944.958984, 128.266784),
     Vec3(1427.594238, 2308.249023, 4.196350)],
    "DefendACatTwoTeammates"
)

if __name__ == "__main__":
    get_nearest_neighbors([defend_a_cat_parameters, defend_a_cat_teammates_behind_parameters])
