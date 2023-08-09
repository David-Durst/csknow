import h5py
import pandas as pd

from learn_bot.latent.engagement.column_names import round_id_column, tick_id_column
from learn_bot.latent.order.column_names import c4_pos_cols
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.load_data import LoadDataResult, LoadDataOptions
from learn_bot.latent.train import train_test_split_file_name
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

load_data_options = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=False,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_all_human_data=True,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=False,
    limit_manual_data_to_only_enemies_no_nav=False,
    limit_by_similarity=False,
    train_test_split_file_name=train_test_split_file_name
)

plant_tick_id_column = 'plant tick id'
test_plant_states_file_name = 'test_plant_states.hdf5'


def create_test_plant_states():
    load_data_result = LoadDataResult(load_data_options)

    cols_to_gets = [round_id_column, tick_id_column] + c4_pos_cols
    alive_cols = []

    for player_place_area_columns in specific_player_place_area_columns:
        cols_to_gets += player_place_area_columns.pos + player_place_area_columns.view_angle + \
                        [player_place_area_columns.alive]
        alive_cols.append(player_place_area_columns.alive)

    test_start_dfs = []
    for hdf5_wrapper in load_data_result.multi_hdf5_wrapper.hdf5_wrappers:
        df = load_hdf5_to_pd(hdf5_wrapper.hdf5_path, cols_to_get=cols_to_gets)
        train_test_split = load_data_result.multi_hdf5_wrapper.train_test_splits[hdf5_wrapper.hdf5_path]
        df[plant_tick_id_column] = df.groupby(round_id_column)[tick_id_column].transform('min')
        start_df = df[df[plant_tick_id_column] == df[tick_id_column]]
        test_start_df = start_df[~start_df[round_id_column].isin(train_test_split.train_group_ids)]
        test_start_dfs.append(test_start_df)

    concat_test_start_df = pd.concat(test_start_dfs)

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

    test_plant_states_path = \
        load_data_result.multi_hdf5_wrapper.train_test_split_path.parent / test_plant_states_file_name
    test_plant_states_hdf5_file = h5py.File(test_plant_states_path, 'w')
    data_group = test_plant_states_hdf5_file.create_group('data')

    for col_name in concat_test_start_df.columns:
        data_group.create_dataset(col_name, data=concat_test_start_df.loc[:, col_name])

    test_plant_states_hdf5_file.close()


if __name__ == "__main__":
    create_test_plant_states()