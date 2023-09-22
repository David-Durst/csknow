from enum import Enum
from typing import List, Optional

import pandas as pd

from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import AABB, data_ticks_per_second


class FilterEventType(Enum):
    Fire = 1
    Kill = 2
    KeyArea = 3


KeyAreas = List[AABB]
seconds_round_key_event = 3


def filter_trajectory_by_key_events(filter_event_type: FilterEventType, trajectory_df: pd.DataFrame,
                                    key_areas: Optional[KeyAreas] = None) -> pd.DataFrame:

    key_event_condition = trajectory_df[round_id_column] != trajectory_df[round_id_column]
    # since this was split with : rather than _, need to remove last _
    for player_place_area_columns in specific_player_place_area_columns:
        if filter_event_type == FilterEventType.Fire:
            key_event_condition = key_event_condition | \
                                  (trajectory_df[player_place_area_columns.player_fire_in_last_5s] <= 0.05)
        elif filter_event_type == FilterEventType.Kill:
            lagged_alive = trajectory_df[player_place_area_columns.alive].shift(periods=1)
            lagged_round_id = trajectory_df[round_id_column].shift(periods=1)
            key_event_condition = key_event_condition | \
                                  (lagged_alive & ~trajectory_df[player_place_area_columns.alive] &
                                   trajectory_df[round_id_column] == lagged_round_id)
        else:
            assert key_areas is not None
            for key_area in key_areas:
                x_condition = (trajectory_df[player_place_area_columns.pos[0]] >= key_area.min.x) & \
                              (trajectory_df[player_place_area_columns.pos[0]] <= key_area.max.x)
                y_condition = (trajectory_df[player_place_area_columns.pos[1]] >= key_area.min.y) & \
                              (trajectory_df[player_place_area_columns.pos[1]] <= key_area.max.y)
                z_condition = (trajectory_df[player_place_area_columns.pos[2]] >= key_area.min.z) & \
                              (trajectory_df[player_place_area_columns.pos[2]] <= key_area.max.z)
                key_event_condition = key_event_condition | (x_condition & y_condition & z_condition)

    # mul by 2 for both directions, add 1 so odd and contain center
    time_extended_condition: pd.Series = key_event_condition \
        .rolling(data_ticks_per_second * seconds_round_key_event * 2 + 1, center=True, min_periods=1) \
        .apply(lambda x: x.any(), raw=True).astype(bool)

    return trajectory_df[time_extended_condition]
