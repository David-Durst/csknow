import h5py
import pandas as pd

from learn_bot.latent.analyze.comparison_column_names import all_human_vs_small_human_similarity_hdf5_data_path, \
    all_human_vs_human_28_similarity_hdf5_data_path, small_human_good_rounds, all_human_28_second_filter_good_rounds
from learn_bot.latent.engagement.column_names import round_id_column, tick_id_column
from learn_bot.latent.order.column_names import c4_pos_cols, team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, get_similarity_column
from learn_bot.latent.place_area.load_data import LoadDataResult, LoadDataOptions
from learn_bot.latent.train_paths import train_test_split_file_name
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd, save_pd_to_hdf5
from learn_bot.libs.multi_hdf5_wrapper import absolute_to_relative_train_test_key

load_data_options = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=False,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_human_28_data=False,
    use_all_human_data=True,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=False,
    limit_manual_data_to_only_enemies_no_nav=False,
    #hand_labeled_push_round_ids=[all_human_28_second_filter_good_rounds],
    #similarity_dfs=[load_hdf5_to_pd(all_human_vs_human_28_similarity_hdf5_data_path)],
    load_similarity_data=True,
    limit_by_similarity=False,
    train_test_split_file_name=train_test_split_file_name
)

hdf5_key_column = 'hdf5 key'
plant_tick_id_column = 'plant tick id'
num_ct_alive_column = 'number CT alive'
num_t_alive_column = 'number CT alive'
all_test_plant_states_file_name = 'test_plant_states.hdf5'
push_only_test_plant_states_file_name = 'push_only_test_plant_states.hdf5'
save_only_test_plant_states_file_name = 'save_only_test_plant_states.hdf5'
filter_for_push = True
filter_for_save = False
filter_for_players_alive = True

def create_test_plant_states():
    load_data_result = LoadDataResult(load_data_options)

    cols_to_gets = [round_id_column, tick_id_column] + c4_pos_cols
    alive_cols = []
    ct_alive_cols = []
    t_alive_cols = []

    for player_place_area_columns in specific_player_place_area_columns:
        cols_to_gets += player_place_area_columns.pos + player_place_area_columns.view_angle + \
                        [player_place_area_columns.alive]
        alive_cols.append(player_place_area_columns.alive)
        if team_strs[0] in player_place_area_columns.alive:
            ct_alive_cols.append(player_place_area_columns.alive)
        else:
            t_alive_cols.append(player_place_area_columns.alive)


    test_start_dfs = []
    total_save_or_push_rounds = 0
    for hdf5_wrapper in load_data_result.multi_hdf5_wrapper.hdf5_wrappers:
        df = load_hdf5_to_pd(hdf5_wrapper.hdf5_path, cols_to_get=cols_to_gets)
        id_df = hdf5_wrapper.id_df
        train_test_split = load_data_result.multi_hdf5_wrapper.train_test_splits[
            absolute_to_relative_train_test_key(hdf5_wrapper.hdf5_path)]
        df[plant_tick_id_column] = df.groupby(round_id_column)[tick_id_column].transform('min')
        df[get_similarity_column(0)] = id_df[get_similarity_column(0)]
        df[hdf5_key_column] = str(absolute_to_relative_train_test_key(hdf5_wrapper.hdf5_path))
        start_df = df[df[plant_tick_id_column] == df[tick_id_column]]
        if filter_for_push:
            start_df = start_df[start_df[get_similarity_column(0)] > 0.5]
        elif filter_for_save:
            start_df = start_df[start_df[get_similarity_column(0)] < 0.5]
        total_save_or_push_rounds += len(start_df)
        test_start_df = start_df[~start_df[round_id_column].isin(train_test_split.train_group_ids)].copy()
        test_start_df[num_ct_alive_column] = test_start_df[ct_alive_cols].sum(axis=1)
        test_start_df[num_t_alive_column] = test_start_df[t_alive_cols].sum(axis=1)
        # 42 has like 20 save rounds, but all fail the 4/3 alive test
        #if '42' in str(hdf5_wrapper.hdf5_path.name):
        #    print('round with no entries in test fitting 4/3 constraint')
        if filter_for_players_alive:
            test_start_df = test_start_df[(test_start_df[num_ct_alive_column] <= 4) &
                                          (test_start_df[num_t_alive_column] <= 3)]
        test_start_dfs.append(test_start_df)

    concat_test_start_df = pd.concat(test_start_dfs)
    concat_test_start_df = concat_test_start_df.sample(frac=1., random_state=42)

    # add all the columns that are legacy from old analyses
    concat_test_start_df.loc[:, "round end tick id"] = -1
    concat_test_start_df.loc[:, "tick length"] = -1
    concat_test_start_df.loc[:, "plant id"] = -1
    concat_test_start_df.loc[:, "defusal id"] = -1
    concat_test_start_df.loc[:, "winner team"] = 0
    concat_test_start_df.loc[:, "c4 defused"] = False

    # cast all alive columns to bools
    for alive_col in alive_cols:
        concat_test_start_df.loc[:, alive_col] = concat_test_start_df.loc[:, alive_col].astype('bool')

    test_plant_states_file_name = all_test_plant_states_file_name
    if filter_for_push:
        test_plant_states_file_name = push_only_test_plant_states_file_name
    elif filter_for_save:
        test_plant_states_file_name = save_only_test_plant_states_file_name
    test_plant_states_path = \
        load_data_result.multi_hdf5_wrapper.train_test_split_path.parent / test_plant_states_file_name
    save_pd_to_hdf5(test_plant_states_path, concat_test_start_df)
    print(f"num test rounds {len(concat_test_start_df)}")


if __name__ == "__main__":
    create_test_plant_states()