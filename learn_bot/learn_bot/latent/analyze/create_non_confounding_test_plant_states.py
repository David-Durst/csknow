import pandas as pd

from learn_bot.latent.analyze.create_test_plant_states import load_data_options, push_only_test_plant_states_file_name, \
    hdf5_key_column
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd, save_pd_to_hdf5

camera_x_column = "camera x"
camera_y_column = "camera y"
camera_z_column = "camera z"
camera_yaw_column = "camera view angle x"
camera_pitch_column = "camera view angle y"
camera_columns = [camera_x_column, camera_y_column, camera_z_column, camera_pitch_column, camera_yaw_column]
non_confound_test_plant_states_file_name = 'non_confound_test_plant_states.hdf5'


def create_non_confounding_test_plant_states():
    load_data_result = LoadDataResult(load_data_options)
    test_plant_states_path = \
        load_data_result.multi_hdf5_wrapper.train_test_split_path.parent / push_only_test_plant_states_file_name
    test_plant_states_df = load_hdf5_to_pd(test_plant_states_path, cast_bool_to_int=False)
    test_plant_states_df[hdf5_key_column] = test_plant_states_df[hdf5_key_column].str.decode("utf-8")
    rounds_without_confounds_path = \
        load_data_result.multi_hdf5_wrapper.train_test_split_path.parent / 'rounds_without_confounds.csv'
    rounds_without_confounds_df = \
        pd.read_csv(rounds_without_confounds_path).loc[:, [hdf5_key_column, round_id_column] + camera_columns]
    non_confounding_test_plant_states = \
        test_plant_states_df.merge(rounds_without_confounds_df, on=[hdf5_key_column, round_id_column])
    non_confounding_test_plant_states_path = \
        load_data_result.multi_hdf5_wrapper.train_test_split_path.parent / non_confound_test_plant_states_file_name
    save_pd_to_hdf5(non_confounding_test_plant_states_path, non_confounding_test_plant_states)


if __name__ == "__main__":
    create_non_confounding_test_plant_states()
