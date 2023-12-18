from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import get_hdf5_to_test_round_ids, \
    get_test_plant_states_pd
from learn_bot.latent.analyze.create_test_plant_states import hdf5_key_column
from learn_bot.latent.engagement.column_names import tick_id_column, round_id_column, game_id_column, \
    round_number_column, game_tick_number_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.order.column_names import team_strs, c4_pos_cols
from learn_bot.latent.place_area.column_names import grenade_columns, grenade_throw_tick_col, grenade_active_tick_col, \
    grenade_expired_tick_col, grenade_destroy_tick_col, specific_player_place_area_columns
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.vis.run_vis_checkpoint import load_data_options
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.libs.pd_printing import set_pd_print_options

demo_file_column = 'demo file'
num_ticks_column = 'num ticks'
plant_state_index_column = 'plant state index'
start_game_tick_length_column = 'start game tick'
game_tick_length_column = 'game tick length'
num_ct_alive_column = 'num ct alive'
num_t_alive_column = 'num t alive'

min_round_seconds = 5
game_tick_rate = 128

def add_num_alive_columns(df: pd.DataFrame):
    ct_alive_cols = []
    t_alive_cols = []
    for player_place_area_columns in specific_player_place_area_columns:
        if team_strs[0] in player_place_area_columns.alive:
            ct_alive_cols.append(player_place_area_columns.alive)
        else:
            t_alive_cols.append(player_place_area_columns.alive)

    df[num_ct_alive_column] = df[ct_alive_cols].sum(axis=1)
    df[num_t_alive_column] = df[t_alive_cols].sum(axis=1)


def find_rounds_without_confounds():
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, use_test_data_only=True)

    set_pd_print_options()
    pd.set_option('display.max_colwidth', None)
    hdf5_to_round_ids = get_hdf5_to_test_round_ids(push_only=True)[1]
    test_plant_states_pd = get_test_plant_states_pd(push_only=True)
    make_index_column(test_plant_states_pd)
    test_plant_states_pd[plant_state_index_column] = test_plant_states_pd['index']
    push_no_grenade_round_dicts: List[Dict] = []

    with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
        for i in range(len(loaded_model.dataset.data_hdf5s)):
            #if i > 0:
            #    break
            loaded_model.cur_hdf5_index = i
            loaded_model.load_cur_dataset_only()

            id_df = loaded_model.get_cur_id_df()

            hdf5_file = loaded_model.dataset.data_hdf5s[i].get_hdf5_path_with_parent_folder()
            push_round_ids = hdf5_to_round_ids[hdf5_file]

            grenades_df = loaded_model.get_cur_extra_df(grenade_columns)
            throw_tick_ids = grenades_df[grenade_throw_tick_col].unique()
            active_tick_ids = grenades_df[grenade_active_tick_col].unique()
            expired_tick_ids = grenades_df[grenade_expired_tick_col].unique()
            destroy_tick_ids = grenades_df[grenade_destroy_tick_col].unique()

            grenade_id_df = id_df[(id_df[tick_id_column].isin(throw_tick_ids)) |
                                  (id_df[tick_id_column].isin(active_tick_ids)) |
                                  (id_df[tick_id_column].isin(expired_tick_ids)) |
                                  (id_df[tick_id_column].isin(destroy_tick_ids))]
            grenade_rounds = grenade_id_df[round_id_column].unique()
            push_no_grenade_all_ticks_in_rounds_id_df = id_df[~id_df[round_id_column].isin(grenade_rounds) &
                                                              id_df[round_id_column].isin(push_round_ids)]
            push_no_grenade_round_summaries_df = push_no_grenade_all_ticks_in_rounds_id_df.groupby(round_id_column).agg({
                game_id_column: 'first',
                round_id_column: 'first',
                round_number_column: 'first',
                game_tick_number_column: ['first', 'last'],
            })

            for _, round_row in push_no_grenade_round_summaries_df.iterrows():
                push_no_grenade_round_dicts.append({
                    'hdf5_index': i,
                    hdf5_key_column: str(hdf5_file),
                    demo_file_column: loaded_model.cur_demo_names[round_row[game_id_column]['first']],
                    round_id_column: round_row[round_id_column]['first'],
                    round_number_column: round_row[round_number_column]['first'],
                    start_game_tick_length_column: round_row[game_tick_number_column]['first'],
                    game_tick_length_column: round_row[game_tick_number_column]['last'] -
                                             round_row[game_tick_number_column]['first'] + 1,
                })

            pbar.update(1)

    result_df = pd.DataFrame.from_records(push_no_grenade_round_dicts)
    result_df = result_df[result_df[game_tick_length_column] > min_round_seconds * game_tick_rate]
    print(len(result_df))
    add_num_alive_columns(test_plant_states_pd)
    plant_states_index_hdf5_df = test_plant_states_pd.loc[:, [hdf5_key_column, round_id_column,
                                                              plant_state_index_column, num_ct_alive_column, num_t_alive_column,
                                                              c4_pos_cols[0]]]
    result_with_plant_state_index_df = result_df.merge(plant_states_index_hdf5_df, on=[hdf5_key_column, round_id_column])
    print(len(result_with_plant_state_index_df))
    randomized_result_df = result_with_plant_state_index_df.sample(frac=1, random_state=42)
    a_four_ct = randomized_result_df[(randomized_result_df[num_ct_alive_column] == 4) &
                                     (randomized_result_df[c4_pos_cols[0]] > 0)].iloc[[0]]
    b_four_ct = randomized_result_df[(randomized_result_df[num_ct_alive_column] == 4) &
                                     (randomized_result_df[c4_pos_cols[0]] < 0)].iloc[[0]]
    a_three_ct_three_t = randomized_result_df[(randomized_result_df[num_ct_alive_column] == 3) &
                                              (randomized_result_df[num_t_alive_column] == 3) &
                                              (randomized_result_df[c4_pos_cols[0]] > 0)].iloc[[0]]
    b_three_ct_three_t = randomized_result_df[(randomized_result_df[num_ct_alive_column] == 3) &
                                              (randomized_result_df[num_t_alive_column] == 3) &
                                              (randomized_result_df[c4_pos_cols[0]] < 0)].iloc[[0]]
    a_two_ct_one_t = randomized_result_df[(randomized_result_df[num_ct_alive_column] == 2) &
                                          (randomized_result_df[num_t_alive_column] == 1) &
                                          (randomized_result_df[c4_pos_cols[0]] > 0)].iloc[[0]]
    b_two_ct_one_t = randomized_result_df[(randomized_result_df[num_ct_alive_column] == 2) &
                                          (randomized_result_df[num_t_alive_column] == 1) &
                                          (randomized_result_df[c4_pos_cols[0]] < 0)].iloc[[0]]
    a_one_ct_two_t = randomized_result_df[(randomized_result_df[num_ct_alive_column] == 1) &
                                          (randomized_result_df[num_t_alive_column] == 2) &
                                          (randomized_result_df[c4_pos_cols[0]] > 0)].iloc[[0]]
    b_one_ct_two_t = randomized_result_df[(randomized_result_df[num_ct_alive_column] == 1) &
                                          (randomized_result_df[num_t_alive_column] == 2) &
                                          (randomized_result_df[c4_pos_cols[0]] < 0)].iloc[[0]]
    selected_result_df = pd.concat([a_four_ct, b_four_ct,
                                    a_three_ct_three_t, b_three_ct_three_t,
                                    a_two_ct_one_t, b_two_ct_one_t,
                                    a_one_ct_two_t, b_one_ct_two_t])
    print(selected_result_df)
    print(selected_result_df.to_csv())
    print("cp " + " ".join(selected_result_df[demo_file_column].tolist()) +
          " ~/.local/share/Steam/steamapps/common/Counter-Strike\\ Global\\ Offensive/csgo/")


if __name__ == "__main__":
    find_rounds_without_confounds()
