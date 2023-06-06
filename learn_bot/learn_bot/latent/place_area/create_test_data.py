import pandas as pd
import numpy as np
from learn_bot.latent.place_area.column_names import *

num_test_ticks = 100

def create_left_right_test_data(all_data_df: pd.DataFrame) -> pd.DataFrame:
    base_series = all_data_df.iloc[0].copy()
    for col in all_data_df.columns:
        if all_data_df[col].dtype.type != np.object_:
            base_series[col] = 0

    base_series['valid'] = 1
    base_series[test_success_col] = 1

    new_ticks: List[pd.Series] = []
    for i in range(num_test_ticks):
        new_tick_series = base_series.copy()
        new_tick_series['id'] = i
        new_tick_series[tick_id_column] = i
        new_tick_series[specific_player_place_area_columns[0].player_id] = 1
        new_tick_series[specific_player_place_area_columns[0].alive] = 1
        new_tick_series[specific_player_place_area_columns[0].ct_team] = 1
        new_tick_series[specific_player_place_area_columns[0].pos[0]] = float(i)
        new_tick_series[c4_plant_a_col] = 1
        new_tick_series[specific_player_place_area_columns[0].delta_pos[0]] = 1
        new_ticks.append(new_tick_series)

    for i in range(num_test_ticks):
        new_tick_series = base_series.copy()
        new_tick_series['id'] = i + num_test_ticks
        new_tick_series[tick_id_column] = i + num_test_ticks
        new_tick_series[specific_player_place_area_columns[0].player_id] = 1
        new_tick_series[specific_player_place_area_columns[0].alive] = 1
        new_tick_series[specific_player_place_area_columns[0].ct_team] = 1
        new_tick_series[specific_player_place_area_columns[0].pos[0]] = float(i)
        new_tick_series[c4_plant_a_col] = 1
        new_tick_series[specific_player_place_area_columns[0].delta_pos[1]] = 1
        new_ticks.append(new_tick_series)

    return pd.DataFrame(new_ticks)
