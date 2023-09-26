from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import List, Optional, Dict

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


@dataclass
class TrajectoryEvents:
    data: List[pd.DataFrame]
    valid_players: List[pd.DataFrame]


def filter_trajectory_by_key_events(filter_event_type: FilterEventType, trajectory_df: pd.DataFrame,
                                    time_extend: bool,
                                    key_areas: Optional[KeyAreas] = None,
                                    key_area_team: KeyAreaTeam = KeyAreaTeam.Both) -> TrajectoryEvents:

    key_event_condition = trajectory_df[round_id_column] != trajectory_df[round_id_column]
    player_conditions: Dict[str, pd.Series] = {}
    for player_place_area_columns in specific_player_place_area_columns:
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
        player_conditions[player_place_area_columns.player_id] = per_player_condition
        key_event_condition = key_event_condition | per_player_condition

    player_conditions_df = pd.DataFrame(player_conditions)
    # if just a single condition, or them all together
    #if filter_event_type in [FilterEventType.Fire, FilterEventType.Kill, FilterEventType.KeyArea]:
    #    key_event_condition = fire_condition | kill_condition | key_area_condition
    ## for and, want anyone firing while a player (of right time) is in the key area
    #else:
    #    key_event_condition = fire_condition & key_area_condition

    # mul by 2 for both directions, add 1 so odd and contain center
    if time_extend:
        time_extended_condition: pd.Series = key_event_condition \
            .rolling(int(ceil(data_ticks_per_second * seconds_round_key_event)) * 2 + 1, center=True, min_periods=1) \
            .apply(lambda x: x.any(), raw=True).astype(bool)
    else:
        time_extended_condition: pd.Series = key_event_condition.astype(bool)

    # split into contiguous chunks
    true_regions = (~time_extended_condition).cumsum()
    true_regions = true_regions[time_extended_condition]
    true_regions_df = pd.DataFrame({'data': true_regions, 'index': true_regions.index})
    true_regions_start_end = true_regions_df.groupby('data').agg({'index': ['min', 'max']})

    # if not time extending, then just looking for individual events, so keep them as one df
    if time_extend:
        data_per_event_dfs: List[pd.DataFrame] = []
        player_conditions_per_event_dfs: List[pd.DataFrame] = []
        for _, true_region in true_regions_start_end['index'].iterrows():
            data_per_event_dfs.append(trajectory_df.loc[true_region['min']:true_region['max']])
            player_conditions_per_event_dfs.append(player_conditions_df.loc[true_region['min']:true_region['max']])
        return TrajectoryEvents(data_per_event_dfs, player_conditions_per_event_dfs)
    else:
        return TrajectoryEvents([trajectory_df[time_extended_condition]],
                                [player_conditions_df[time_extended_condition]])

