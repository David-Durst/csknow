import pandas as pd

from learn_bot.latent.place_area.column_names import *
from learn_bot.latent.place_area.pos_abs_delta_conversion import *

num_alive_col = 'num alive'


def filter_region(all_data_df: pd.DataFrame, valid_range: AABB, must_plant_a: bool, must_plant_b: bool,
                  required_num_alive: List[int]) -> pd.DataFrame:
    # build or because allow any player to be in range
    filter_builder: pd.Series = all_data_df[tick_id_column] != all_data_df[tick_id_column]
    for player_columns in specific_player_place_area_columns:
        filter_builder = filter_builder | \
                         (all_data_df[player_columns.pos[0]].between(valid_range.min.x, valid_range.max.x) &
                          all_data_df[player_columns.pos[1]].between(valid_range.min.y, valid_range.max.y))
    if must_plant_a:
        filter_builder = filter_builder & all_data_df[c4_plant_a_col]
    if must_plant_b:
        filter_builder = filter_builder & all_data_df[c4_plant_b_col]
    pos_restricted_df = all_data_df[filter_builder].copy()

    # after big cut by pos, restrict next by num alive
    alive_cols = []
    for player_columns in specific_player_place_area_columns:
        alive_cols.append(player_columns.alive)
    pos_restricted_df[num_alive_col] = pos_restricted_df[alive_cols].sum(axis=1)
    pos_alive_restricted_df = pos_restricted_df[pos_restricted_df[num_alive_col].isin(required_num_alive)].copy()

    return pos_alive_restricted_df

