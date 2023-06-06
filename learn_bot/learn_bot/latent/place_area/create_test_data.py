import pandas as pd
import numpy as np
from learn_bot.latent.place_area.column_names import *

num_test_ticks = 100


def add_tick_row(id: int, player_id: int, alive: int, ct_team: int, pos_x: float, pos_y: float,
                 planted_a: int, delta_pos_index: int, base_series: pd.Series, new_ticks: List[pd.Series]):
    new_tick_series = base_series.copy()
    new_tick_series['id'] = id
    new_tick_series[tick_id_column] = id
    new_tick_series[specific_player_place_area_columns[0].player_id] = player_id
    new_tick_series[specific_player_place_area_columns[0].alive] = alive
    new_tick_series[specific_player_place_area_columns[0].ct_team] = ct_team
    new_tick_series[specific_player_place_area_columns[0].pos[0]] = pos_x
    new_tick_series[specific_player_place_area_columns[0].pos[1]] = pos_y
    new_tick_series[c4_plant_a_col] = planted_a
    new_tick_series[specific_player_place_area_columns[0].delta_pos[delta_pos_index]] = 1
    new_ticks.append(new_tick_series)


def create_left_right_train_data(all_data_df: pd.DataFrame) -> pd.DataFrame:
    base_series = all_data_df.iloc[0].copy()
    for col in all_data_df.columns:
        if all_data_df[col].dtype.type != np.object_:
            base_series[col] = 0

    base_series['valid'] = 1
    base_series[test_success_col] = 1

    new_ticks: List[pd.Series] = []
    # increasing x and
    for i in range(num_test_ticks):
        add_tick_row(i, 1, 1, 1, float(i), 0., 1, 0, base_series, new_ticks)

    for i in range(num_test_ticks):
        add_tick_row(i, 1, 1, 1, float(i), 1., 1, 0, base_series, new_ticks)

    for i in range(num_test_ticks):
        add_tick_row(i, 1, 1, 1, float(i), 0., 0, -1, base_series, new_ticks)

    for i in range(num_test_ticks):
        add_tick_row(i, 1, 1, 1, float(i), 1., 0, -1, base_series, new_ticks)

    return pd.DataFrame(new_ticks)


def create_left_right_test_data(all_data_df: pd.DataFrame) -> pd.DataFrame:
    base_series = all_data_df.iloc[0].copy()
    for col in all_data_df.columns:
        if all_data_df[col].dtype.type != np.object_:
            base_series[col] = 0

    base_series['valid'] = 1
    base_series[test_success_col] = 1

    new_ticks: List[pd.Series] = []
    # increasing x and
    for i in range(num_test_ticks):
        add_tick_row(i, 1, 1, 1, -20 * float(i), 4., 1, -1, base_series, new_ticks)

    for i in range(num_test_ticks):
        add_tick_row(i, 1, 1, 1, -20 * float(i), 4., 0, -1, base_series, new_ticks)

    return pd.DataFrame(new_ticks)
