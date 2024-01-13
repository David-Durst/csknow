from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import get_hdf5_to_test_round_ids, \
    get_test_plant_states_pd
from learn_bot.latent.analyze.create_test_plant_states import hdf5_key_column
from learn_bot.latent.analyze.find_rounds_without_confounds import add_num_alive_columns
from learn_bot.latent.engagement.column_names import tick_id_column, round_id_column, game_id_column, \
    round_number_column, game_tick_number_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.order.column_names import team_strs, c4_pos_cols
from learn_bot.latent.place_area.column_names import grenade_columns, grenade_throw_tick_col, grenade_active_tick_col, \
    grenade_expired_tick_col, grenade_destroy_tick_col, specific_player_place_area_columns
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.vis.run_vis_checkpoint import load_data_options
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.libs.hdf5_to_pd import save_pd_to_hdf5
from learn_bot.libs.multi_hdf5_wrapper import train_test_split_folder_path
from learn_bot.libs.pd_printing import set_pd_print_options

remix_push_only_test_plant_states_file_name = 'remix_push_only_test_plant_states.hdf5'

num_defensive_setups = 2
num_offensive_setups = 4
planted_a_col = "planted a"

def remix_test_plant_rounds():
    test_plant_states_pd = get_test_plant_states_pd(push_only=True).copy()
    test_plant_states_pd[planted_a_col] = test_plant_states_pd[c4_pos_cols[0]] > 0
    add_num_alive_columns(test_plant_states_pd)

    pd.set_option('display.max_colwidth', None)
    ct_columns = [c for c in test_plant_states_pd.columns if ' CT ' in c or ' ct ' in c]
    #t_columns = [c for c in test_plant_states_pd.columns if ' T ' in c or ' t ' in c]

    defensive_df = test_plant_states_pd.sample(n=num_defensive_setups, random_state=43)

    defensive_df = defensive_df.loc[defensive_df.index.repeat(num_offensive_setups)]
    defensive_df.reset_index(inplace=True)

    offensive_planted_a_df = test_plant_states_pd[test_plant_states_pd[planted_a_col]].sample(
        n=num_offensive_setups, random_state=84)
    offensive_planted_a_df.reset_index(inplace=True)
    offensive_planted_b_df = test_plant_states_pd[~test_plant_states_pd[planted_a_col]].sample(
        n=num_offensive_setups, random_state=84)
    offensive_planted_b_df.reset_index(inplace=True)

    for idx, defensive_row in defensive_df.iterrows():
        if defensive_row[planted_a_col]:
            offensive_df = offensive_planted_a_df
        else:
            offensive_df = offensive_planted_b_df
        offensive_idx = idx % len(offensive_df)
        defensive_df.loc[idx, ct_columns] = offensive_df.loc[offensive_idx, ct_columns]

    alive_cols = []
    helmet_cols = []
    for player_place_area_columns in specific_player_place_area_columns:
        alive_cols.append(player_place_area_columns.alive)
        helmet_cols.append(player_place_area_columns.player_helmet)
    for alive_col in alive_cols:
        defensive_df.loc[:, alive_col] = defensive_df.loc[:, alive_col].astype('bool')
    for helmet_col in helmet_cols:
        defensive_df.loc[:, helmet_col] = defensive_df.loc[:, helmet_col].astype('bool')
    defensive_df.loc[:, "c4 defused"] = False

    save_pd_to_hdf5(train_test_split_folder_path / remix_push_only_test_plant_states_file_name,
                    defensive_df)


if __name__ == "__main__":
    remix_test_plant_rounds()
