import dataclasses
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_and_events import plot_trajectory_dfs_and_event
from learn_bot.latent.analyze.compare_trajectories.region_constraints.compute_constraint_metrics import check_constraint_metrics
from learn_bot.latent.analyze.compare_trajectories.run_trajectory_comparison import all_human_load_data_option, \
    all_human_vs_all_human_config
from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path
from learn_bot.latent.analyze.knn.select_alive_players import PlayerColumnIndices
from learn_bot.latent.engagement.column_names import round_id_column, row_id_column, game_tick_number_column, \
    index_column
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns

l2_distance_col = 'l2 distance'
hdf5_id_col = 'hdf5 id'
target_full_table_id_col = 'target full table id'
player_to_full_table_counter_col = 'player to full table counter'
max_game_tick_number_column = 'max ' + game_tick_number_column
max_index_column = 'max ' + index_column


# note: this id_df is only for rows in all_np, which may not be all rows in data set
def all_np_to_pos_alive_df(all_np: np.ndarray, id_df: pd.DataFrame, hdf5_id: int,
                           loaded_model: LoadedModel) -> pd.DataFrame:
    player_column_indices = PlayerColumnIndices(loaded_model.model)

    column_names: List[str] = []
    column_indices: List[int] = []
    for i, player_place_area_columns in enumerate(specific_player_place_area_columns):
        column_names.append(player_place_area_columns.alive)
        column_indices.append(player_column_indices.alive_cols[i])
        for d in range(3):
            column_names.append(player_place_area_columns.pos[d])
            column_indices.append(player_column_indices.pos_cols[i][d])

    pos_alive_only_df = pd.DataFrame(all_np[:, column_indices], columns=column_names)
    pos_alive_df = pd.concat([pos_alive_only_df, id_df.reset_index()], axis=1)
    pos_alive_df[hdf5_id_col] = hdf5_id
    return pos_alive_df


game_tick_rate = 128
max_future_seconds = 10


def collect_plot_plot_min_distance_rounds(loaded_model: LoadedModel, min_distance_rounds_df: pd.DataFrame, test_name: str,
                                          restrict_future: Optional[bool], num_matches: int, plot: bool,
                                          num_future_ticks: Optional[int]) -> List[np.ndarray]:
    # can't just append to this lsit, order of hdf5's might not match order in which they are used,
    # need to sort list by order of appearing in min_distance_rounds_df
    min_distance_nps: List[np.ndarray] = [None for i in range(len(min_distance_rounds_df))]
    min_distance_pos_dfs: List[pd.DataFrame] = []
    target_full_table_ids: List[int] = []
    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        min_distance_rounds_cur_hdf5 = min_distance_rounds_df[min_distance_rounds_df[hdf5_id_col] == i]
        for _, round_row in min_distance_rounds_cur_hdf5.iterrows():
            #if i != 18 or round_row[round_id_column] != 489:
            #    continue
            min_distance_condition = (hdf5_wrapper.id_df[round_id_column] == round_row[round_id_column]) & \
                                     (hdf5_wrapper.id_df[row_id_column] >= round_row[row_id_column])
            if restrict_future is not None and restrict_future:
                if num_future_ticks is not None:
                    min_distance_condition = min_distance_condition & \
                                             (hdf5_wrapper.id_df[index_column] <
                                              round_row[index_column] + num_future_ticks)
                else:
                    min_distance_condition = min_distance_condition & \
                                             (hdf5_wrapper.id_df[game_tick_number_column] <=
                                              round_row[game_tick_number_column] + game_tick_rate*max_future_seconds)
            min_distance_id_df = hdf5_wrapper.id_df[min_distance_condition]
            # need to use condtiion, not ids, might have filtered down based on id or test/train split
            min_distance_np = hdf5_wrapper.get_all_input_data()[min_distance_condition]
            min_distance_nps[round_row.name] = min_distance_np
            min_distance_pos_dfs.append(all_np_to_pos_alive_df(min_distance_np, min_distance_id_df, i, loaded_model))
            target_full_table_ids.append(round_row[target_full_table_id_col])

    if plot:
        load_data_option = dataclasses.replace(all_human_load_data_option,
                                               custom_rollout_extension="human_knn")
        config = dataclasses.replace(all_human_vs_all_human_config,
                                     predicted_load_data_options=load_data_option,
                                     metric_cost_title=f"Human KNN {test_name}")
        option_str = f"matches_{num_matches}"
        if restrict_future is not None:
            option_str += f"_restrict_future_{restrict_future}"
        plots_path = similarity_plots_path / load_data_option.custom_rollout_extension / option_str
        os.makedirs(plots_path, exist_ok=True)
        restrict_future_str = ""
        if restrict_future is not None:
            restrict_future_str = f"_r_{restrict_future}"
        img = plot_trajectory_dfs_and_event(min_distance_pos_dfs, config, True, True, True, plot_starts=True,
                                      only_plot_post_start=target_full_table_ids)
        constraint_result = check_constraint_metrics(min_distance_pos_dfs, test_name, target_full_table_ids, img)
        img.save(plots_path / f'{test_name}{restrict_future_str}.png')
        constraint_result.save(plots_path / f'{test_name}{restrict_future_str}.csv', 'human knn')
    else:
        return min_distance_nps


