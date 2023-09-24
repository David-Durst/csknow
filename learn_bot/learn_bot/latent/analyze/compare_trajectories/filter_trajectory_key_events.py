from enum import Enum
from math import ceil
from typing import List, Optional

import pandas as pd

from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import AABB, data_ticks_per_second


class FilterEventType(Enum):
    Fire = 1
    Kill = 2
    KeyArea = 3
    FireAndKeyArea = 4

    def __str__(self) -> str:
        if self == FilterEventType.Fire:
            return "Fire"
        elif self == FilterEventType.Kill:
            return "Kill"
        elif self == FilterEventType.KeyArea:
            return "KeyArea"
        else:
            return "FireAndKeyArea"


KeyAreas = List[AABB]
seconds_round_key_event = 0.2


class KeyAreaTeam(Enum):
    CT = 1
    T = 2
    Both = 3


def filter_trajectory_by_key_events(filter_event_type: FilterEventType, trajectory_df: pd.DataFrame,
                                    key_areas: Optional[KeyAreas] = None,
                                    key_area_team: KeyAreaTeam = KeyAreaTeam.Both) -> List[pd.DataFrame]:

    key_event_condition = trajectory_df[round_id_column] != trajectory_df[round_id_column]
    # since this was split with : rather than _, need to remove last _
    # want someone firing, and player on right team in the key area
    for player_place_area_columns in specific_player_place_area_columns:
        # allow anding conditions per player
        per_player_condition = trajectory_df[round_id_column] == trajectory_df[round_id_column]
        if filter_event_type == FilterEventType.Fire or filter_event_type == FilterEventType.FireAndKeyArea:
            per_player_condition = per_player_condition & \
                                   (trajectory_df[player_place_area_columns.player_fire_in_last_5s] <= 0.05)
        if filter_event_type == FilterEventType.Kill:
            lagged_alive = trajectory_df[player_place_area_columns.alive].shift(periods=1)
            per_player_condition = per_player_condition & \
                                   (lagged_alive & ~trajectory_df[player_place_area_columns.alive])
        if filter_event_type == FilterEventType.KeyArea or filter_event_type == FilterEventType.FireAndKeyArea:
            is_ct_player = team_strs[0] in player_place_area_columns.player_id
            assert key_areas is not None
            if (key_area_team == KeyAreaTeam.Both) or \
                    (key_area_team == KeyAreaTeam.CT and is_ct_player) or (
                    key_area_team == KeyAreaTeam.T and not is_ct_player):
                for key_area in key_areas:
                    x_condition = (trajectory_df[player_place_area_columns.pos[0]] >= key_area.min.x) & \
                                  (trajectory_df[player_place_area_columns.pos[0]] <= key_area.max.x)
                    y_condition = (trajectory_df[player_place_area_columns.pos[1]] >= key_area.min.y) & \
                                  (trajectory_df[player_place_area_columns.pos[1]] <= key_area.max.y)
                    z_condition = (trajectory_df[player_place_area_columns.pos[2]] >= key_area.min.z) & \
                                  (trajectory_df[player_place_area_columns.pos[2]] <= key_area.max.z)
                    per_player_condition = per_player_condition & (x_condition & y_condition & z_condition)
        key_event_condition = key_event_condition | per_player_condition

    # mul by 2 for both directions, add 1 so odd and contain center
    time_extended_condition: pd.Series = key_event_condition \
        .rolling(int(ceil(data_ticks_per_second * seconds_round_key_event)) * 2 + 1, center=True, min_periods=1) \
        .apply(lambda x: x.any(), raw=True).astype(bool)

    # split into contiguous chunks
    true_regions = (~time_extended_condition).cumsum()
    true_regions = true_regions[time_extended_condition]
    true_regions_df = pd.DataFrame({'data': true_regions, 'index': true_regions.index})
    true_regions_start_end = true_regions_df.groupby('data').agg({'index': ['min', 'max']})

    result: List[pd.DataFrame] = []
    for _, true_region in true_regions_start_end['index'].iterrows():
        result.append(trajectory_df.loc[true_region['min']:true_region['max']])

    return result
