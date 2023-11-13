import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import math

from learn_bot.latent.analyze.knn.generate_player_index_mappings import generate_all_player_to_column_mappings
from learn_bot.latent.analyze.knn.plot_min_distance_rounds import collect_plot_plot_min_distance_rounds, \
    l2_distance_col, \
    hdf5_id_col, target_full_table_id_col, max_game_tick_number_column, game_tick_rate, max_index_column
from learn_bot.latent.analyze.knn.select_alive_players import get_id_df_and_alive_pos_and_full_table_id_np
from learn_bot.latent.engagement.column_names import round_id_column, game_tick_number_column, index_column
from learn_bot.latent.load_model import load_model_file, LoadedModel
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.train import load_data_options
from learn_bot.libs.vec import Vec3


@dataclass
class PositionSituationParameters:
    ct_pos: List[Vec3]
    t_pos: List[Vec3]
    target_ct: bool
    target_player_index_on_team: int
    name: str

    def get_target_player_index(self):
        if self.target_ct:
            return self.target_player_index_on_team
        else:
            return self.target_player_index_on_team + len(self.ct_pos)


def get_nearest_neighbors(situations: List[PositionSituationParameters], num_matches: int = 20) -> pd.DataFrame:
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, load_pd=False)
    load_data_result.multi_hdf5_wrapper.create_np_arrays(loaded_model.model.cts)

    for situation in situations:
        print(f"processing {situation.name}")
        get_nearest_neighbors_one_situation(situation.ct_pos, situation.t_pos, num_matches,
                                            loaded_model, situation.name, situation.get_target_player_index(), True,
                                            True, None)


@dataclass
class CachedNNData:
    id_dfs: List[pd.DataFrame] = field(default_factory=list)
    alive_pos_nps: List[np.ndarray] = field(default_factory=list)
    full_table_id_nps: List[np.ndarray] = field(default_factory=list)


# if num_future_ticks is None, then will restrict based on game ticks (actual in game clock time) rather than number
# of ticks
def get_nearest_neighbors_one_situation(ct_pos: List[Vec3], t_pos: List[Vec3], num_matches: int,
                                        loaded_model: LoadedModel, situation_name: str, target_player_index: int,
                                        plot: bool, push_only: bool, num_future_ticks: Optional[int],
                                        cached_nn_data: Optional[CachedNNData] = None,
                                        math_decimation: Optional[int] = None) -> Tuple[List[np.ndarray], CachedNNData]:
    start = time.time()
    num_ct_alive = len(ct_pos)
    num_t_alive = len(t_pos)

    players_pos = ct_pos + t_pos

    start_mappings_time = time.time()
    player_to_column_mappings = generate_all_player_to_column_mappings(num_ct_alive, num_t_alive)
    mappings_time = time.time() - start_mappings_time

    min_distance_rounds_per_hdf5: List[pd.DataFrame] = []

    data_time: float = 0.
    math_time: float = 0.
    pd_time: float = 0.

    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        get_data_start_time = time.time()
        if cached_nn_data is None or len(cached_nn_data.id_dfs) <= i:
            id_df, alive_pos_np, full_table_id_np = get_id_df_and_alive_pos_and_full_table_id_np(hdf5_wrapper,
                                                                                                 loaded_model.model,
                                                                                                 num_ct_alive, num_t_alive,
                                                                                                 push_only)
            if cached_nn_data is not None:
                cached_nn_data.id_dfs.append(id_df)
                cached_nn_data.alive_pos_nps.append(alive_pos_np)
                cached_nn_data.full_table_id_nps.append(full_table_id_np)
        else:
            id_df = cached_nn_data.id_dfs[i]
            alive_pos_np = cached_nn_data.alive_pos_nps[i]
            full_table_id_np = cached_nn_data.full_table_id_nps[i]
        data_time += time.time() - get_data_start_time

        if id_df.empty:
            continue

        alive_pos_np = alive_pos_np.astype(np.float32)
        if math_decimation is not None:
            alive_pos_np = alive_pos_np[::math_decimation]
            full_table_id_np = full_table_id_np[::math_decimation]

        base_point_np = np.zeros_like(alive_pos_np)
        for j, player_pos in enumerate(players_pos):
            base_point_np[:, j, 0] = player_pos.x
            base_point_np[:, j, 1] = player_pos.y
            base_point_np[:, j, 2] = player_pos.z

        # find min distance from each point to the base point across all player mappings
        start_math_time = time.time()
        min_euclidean_distance_per_row = np.zeros(alive_pos_np.shape[0])
        min_euclidean_distance_per_row[:] = math.inf
        min_manhattan_distance_per_row = np.zeros(alive_pos_np.shape[0])
        min_manhattan_distance_per_row[:] = math.inf
        target_full_table_id_per_row = np.zeros(alive_pos_np.shape[0], dtype=np.int32)
        for player_to_column_mapping in player_to_column_mappings:
            # sum distance per player for this mapping
            player_indices = []
            column_indices = []
            target_column_index = -1
            for player_index, column_index in player_to_column_mapping.player_to_column.items():
                player_indices.append(player_index)
                column_indices.append(column_index)
                if player_index == target_player_index:
                    target_column_index = column_index
            # use euclidean distance to find most similar global alignment
            mapping_euclidean_distance_per_player = (
                    (alive_pos_np[:, column_indices, 0] - base_point_np[:, player_indices, 0]) ** 2. +
                    (alive_pos_np[:, column_indices, 1] - base_point_np[:, player_indices, 1]) ** 2. +
                    (alive_pos_np[:, column_indices, 2] - base_point_np[:, player_indices, 2]) ** 2.
            ) ** .5
            mapping_euclidean_distance = np.sum(mapping_euclidean_distance_per_player, axis=1)
            min_euclidean_distance_per_row = np.where(mapping_euclidean_distance < min_euclidean_distance_per_row,
                                                      mapping_euclidean_distance, min_euclidean_distance_per_row)
            # use manhattan distance to find which player is target, as that global misalignments
            # may sacrifice target accuracy for overall accuracy
            # WHY DOESN'T THIS DEPEND ON EUCLIDEAN? COULDN'T THIS LEAD TO PLAYER MATCHES FOR DIFFERENT GAME STATES?
            # NO - THIS IS COMPUTES BEST MAPPING ON A PER GAME STATE BASIS. AFTER USING L2 TO COMPUTE BEST POSITION,
            # THIS WILL TELL YOU ALIGNMENT, SO THIS ALWAYS USED AFTER L2 (SORT BY L2 BELOW DONE FIRST, THEN LOOK AT
            # target_full_table_id_col)
            mapping_manhattan_distance_per_player = (
                    np.abs(alive_pos_np[:, column_indices, 0] - base_point_np[:, player_indices, 0]) +
                    np.abs(alive_pos_np[:, column_indices, 1] - base_point_np[:, player_indices, 1]) +
                    np.abs(alive_pos_np[:, column_indices, 2] - base_point_np[:, player_indices, 2])
            )
            mapping_manhattan_distance = np.sum(mapping_manhattan_distance_per_player, axis=1)
            target_full_table_id_per_row = np.where(mapping_manhattan_distance < min_manhattan_distance_per_row,
                                                    full_table_id_np[:, target_column_index], target_full_table_id_per_row)
            min_manhattan_distance_per_row = np.where(mapping_manhattan_distance < min_manhattan_distance_per_row,
                                                      mapping_manhattan_distance, min_manhattan_distance_per_row)
        if math_decimation is not None:
            min_euclidean_distance_per_row = min_euclidean_distance_per_row[:, np.newaxis].repeat(math_decimation, 1) \
                .reshape([-1])
            target_full_table_id_per_row = target_full_table_id_per_row[:, np.newaxis].repeat(math_decimation, 1) \
                .reshape([-1])
        math_time += time.time() - start_math_time

        id_with_distance_df = id_df.copy()
        id_with_distance_df[l2_distance_col] = min_euclidean_distance_per_row
        id_with_distance_df[target_full_table_id_col] = target_full_table_id_per_row
        id_with_distance_df[hdf5_id_col] = i

        start_pd_time = time.time()
        # ensure at least 5 seconds of gameplay to track
        if num_future_ticks is None:
            round_and_max_game_tick_number = id_with_distance_df \
                .groupby(round_id_column, as_index=False)[game_tick_number_column].max() \
                .rename({game_tick_number_column: max_game_tick_number_column}, axis=1)
            id_with_distance_max_tick = id_with_distance_df.merge(round_and_max_game_tick_number, how="left",
                                                                  on=round_id_column)
            id_with_distance_time_limited_tick = \
                id_with_distance_max_tick[(id_with_distance_max_tick[max_game_tick_number_column] -
                                           id_with_distance_max_tick[game_tick_number_column]) > game_tick_rate * 5]
        else:
            round_and_max_index_number = id_with_distance_df \
                .groupby(round_id_column, as_index=False)[index_column].max() \
                .rename({index_column: max_index_column}, axis=1)
            id_with_distance_max_tick = id_with_distance_df.merge(round_and_max_index_number, how="left",
                                                                  on=round_id_column)
            id_with_distance_time_limited_tick = \
                id_with_distance_max_tick[(id_with_distance_max_tick[max_index_column] -
                                           id_with_distance_max_tick[index_column]) > num_future_ticks]
        # sort by distance within round, then take first row to get best match
        id_sorted_by_round_distance_df = \
            id_with_distance_time_limited_tick.sort_values([round_id_column, l2_distance_col])
        min_distance_per_round_df = id_sorted_by_round_distance_df.groupby(round_id_column, as_index=False) \
            .first().sort_values(l2_distance_col).iloc[:num_matches]
        min_distance_rounds_per_hdf5.append(min_distance_per_round_df)
        pd_time += time.time() - start_pd_time
        #if i == 18:
        #    print(min_distance_per_round_df[[hdf5_id_col, round_id_column, l2_distance_col]])
        #    print('breakpoint')

    min_distance_rounds_df = pd.concat(min_distance_rounds_per_hdf5).sort_values(l2_distance_col).iloc[:num_matches]
    #print(f"round id: {min_distance_rounds_df['round id'].iloc[0]}, hdf5 id: {min_distance_rounds_df['hdf5 id'].iloc[0]}")
    #plot_min_distance_rounds(loaded_model, min_distance_rounds_df, situation_name, None, num_matches)
    if plot:
        collect_plot_plot_min_distance_rounds(loaded_model, min_distance_rounds_df, situation_name,
                                              True, num_matches, True, num_future_ticks)
        return collect_plot_plot_min_distance_rounds(loaded_model, min_distance_rounds_df, situation_name,
                                                     False, num_matches, True, num_future_ticks), cached_nn_data
    else:
        start_collect_time = time.time()
        result = collect_plot_plot_min_distance_rounds(loaded_model, min_distance_rounds_df, situation_name,
                                                     True, num_matches, False, num_future_ticks)
        collect_time = time.time() - start_collect_time
        end = time.time()
        print(f"total time {end - start}, mappings time {mappings_time}, data time {data_time}, math time {math_time}, pd time {pd_time}, collect time {collect_time}")
        return result, cached_nn_data


attack_a_spawn_t_long = PositionSituationParameters(
    [Vec3(1430.616699, 1816.052490, -10.300033)],
    [Vec3(1704.018188, 1011.443786, 2.233371)],
    True, 0, "AttackASpawnTLong"
)
attack_a_spawn_t_long_two_teammates = PositionSituationParameters(
    [Vec3(1430.616699, 1816.052490, -10.300033), Vec3(1430.616699, 1516.052490, -10.300033),
     Vec3(1430.616699, 1316.052490, -10.300033)],
    [Vec3(1704.018188, 1011.443786, 2.233371)],
    True, 0, "AttackASpawnTLongTwoTeammates"
)
attack_a_spawn_t_extended_a = PositionSituationParameters(
    [Vec3(1430.616699, 1816.052490, -10.300033)],
    [Vec3(563.968750, 2759.416259, 97.259826)],
    True, 0, "AttackASpawnTExtendedA"
)
attack_b_hole_teammate_b_doors = PositionSituationParameters(
    [Vec3(-550.731201, 2076.939208, -118.991142), Vec3(-1396.848022, 2144.354980, 1.107921)],
    [Vec3(-1879.674072, 2378.484130, 8.714675)],
    True, 0, "AttackBHoleTeammateBDoors"
)
attack_b_doors_teammate_b_hole = PositionSituationParameters(
    [Vec3(-550.731201, 2076.939208, -118.991142), Vec3(-1395.869873, 2652.096679, 125.027893)],
    [Vec3(-1879.674072, 2378.484130, 8.714675)],
    True, 0, "AttackBDoorsTeammateHole"
)
defend_a_cat = PositionSituationParameters(
    [Vec3(563.968750, 2763.999511, 97.379516), Vec3(357.684234, 1650.239990, 27.671302)],
    [Vec3(1160.000976, 2573.304931, 96.338958)],
    False, 0, "DefendACTCat"
)
defend_a_cat_two_teammates = PositionSituationParameters(
    [Vec3(563.968750, 2763.999511, 97.379516), Vec3(357.684234, 1650.239990, 27.671302)],
    [Vec3(1160.000976, 2573.304931, 96.338958), Vec3(1175.846923, 2944.958984, 128.266784),
     Vec3(1427.594238, 2308.249023, 4.196350)],
    False, 0, "DefendACTCatTwoTeammates"
)
defend_a_ct_long = PositionSituationParameters(
    [Vec3(1393.406738, 521.030822, -94.765136), Vec3(1266.489990, 1308.994018, 0.008083)],
    [Vec3(1160.000976, 2573.304931, 96.338958)],
    False, 0, "DefendACTLong"
)
defend_a_ct_long_with_teammate = PositionSituationParameters(
    [Vec3(1393.406738, 521.030822, -94.765136), Vec3(1266.489990, 1308.994018, 0.008083)],
    [Vec3(1160.000976, 2573.304931, 96.338958), Vec3(563.968750, 2763.999511, 97.379516)],
    False, 0, "DefendACTLongWithTeammate"
)
defend_a_ct_long_with_two_teammates = PositionSituationParameters(
    [Vec3(1393.406738, 521.030822, -94.765136), Vec3(1266.489990, 1308.994018, 0.008083)],
    [Vec3(1160.000976, 2573.304931, 96.338958), Vec3(563.968750, 2763.999511, 97.379516),
     Vec3(462.430969, 2006.059082, 133.031250)],
    False, 0, "DefendACTLongWithTwoTeammates"
)
defend_b_ct_site = PositionSituationParameters(
    [Vec3(-1445.885375, 2497.657958, 1.294036)],
    [Vec3(-1977.860229, 1665.813110, 31.853256)],
    False, 0, "DefendBCTSite"
)
defend_b_ct_tuns = PositionSituationParameters(
    [Vec3(-1078.543823, 1232.906372, -87.452003)],
    [Vec3(-1977.860229, 1665.813110, 31.853256)],
    False, 0, "DefendBCTTuns"
)
defend_b_ct_hole = PositionSituationParameters(
    [Vec3(-1179.737426, 2664.458007, 79.098220)],
    [Vec3(-1430.002441, 2676.153564, 16.374132)],
    False, 0, "DefendBCTHole"
)
defend_b_ct_hole_two_teammates = PositionSituationParameters(
    [Vec3(-1179.737426, 2664.458007, 79.098220)],
    [Vec3(-1430.002441, 2676.153564, 16.374132), Vec3(-1925.693725, 2991.133300, 36.464263),
     Vec3(-1925.693725, 2991.133300, 36.464263)],
    False, 0, "DefendBCTHoleTwoTeammates"
)

if __name__ == "__main__":
    #get_nearest_neighbors([attack_a_spawn_t_long_two_teammates], num_matches=5)
    #get_nearest_neighbors([attack_b_hole_teammate_b_doors, attack_b_doors_teammate_b_hole], num_matches=5)
    #get_nearest_neighbors([defend_b_ct_hole], num_matches=100)
    #get_nearest_neighbors([defend_a_ct_long], num_matches=10)
    #exit(0)
    for num_matches in [1, 5, 10, 20, 50, 100]:
        get_nearest_neighbors([attack_a_spawn_t_long, attack_a_spawn_t_long_two_teammates, attack_a_spawn_t_extended_a,
                               attack_b_hole_teammate_b_doors, attack_b_doors_teammate_b_hole, defend_a_cat,
                               defend_a_cat_two_teammates, defend_a_ct_long, defend_a_ct_long_with_teammate,
                               defend_a_ct_long_with_two_teammates, defend_b_ct_site, defend_b_ct_tuns, defend_b_ct_hole,
                               defend_b_ct_hole_two_teammates], num_matches)

