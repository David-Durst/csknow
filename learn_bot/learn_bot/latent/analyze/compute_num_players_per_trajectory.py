from pathlib import Path
from typing import Dict

from learn_bot.latent.engagement.column_names import round_id_column, index_column
from learn_bot.latent.load_model import load_model_file, LoadedModel
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.place_area.push_save_label import PushSaveRoundLabels
from learn_bot.latent.train_paths import default_save_push_round_labels_path
from learn_bot.latent.vis.run_vis_checkpoint import load_data_options
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper


def generate_num_alive_options() -> Dict[int, Dict[int, int]]:
    result = {}
    for num_ct_alive in range(1, 6):
        result[num_ct_alive] = {}
        for num_t_alive in range(1, 6):
            result[num_ct_alive][num_t_alive] = 0
    return result


def print_num_alive_missing(num_alive_counts: Dict[int, Dict[int, int]]):
    for num_ct_alive in range(1, 6):
        for num_t_alive in range(1, 6):
            if num_alive_counts[num_ct_alive][num_t_alive] == 0:
                print(f'missing ct {num_ct_alive}, t {num_t_alive}')


def compute_num_players_per_trajectory_one_hdf5_wrapper(loaded_model: LoadedModel, hdf5_wrapper: HDF5Wrapper,
                                                        hdf5_path: Path, train: bool):
    round_ids_and_row_indices = hdf5_wrapper.id_df.groupby(round_id_column, as_index=False)['id'].first()
    num_alive_options = generate_num_alive_options()

    #if 84 in list(round_ids_and_row_indices[round_id_column]):
    #    print('have 84')

    for _, row in round_ids_and_row_indices.iterrows():
        num_ct_alive = 0
        num_t_alive = 0
        input_data = hdf5_wrapper.get_input_data(row['id'])
        for i, alive_column in enumerate(loaded_model.model.alive_columns):
            if input_data[alive_column]:
                if i < 5:
                    num_ct_alive += 1
                else:
                    num_t_alive += 1
        num_alive_options[num_ct_alive][num_t_alive] += 1

        if '28' in str(hdf5_path) and train and num_ct_alive == 4 and num_t_alive == 2:
            print(f'looking for round {row[round_id_column]}')

    print_num_alive_missing(num_alive_options)


def compute_num_players_per_trajectory(loaded_model: LoadedModel, load_data_result: LoadDataResult):
    for i in range(len(load_data_result.multi_hdf5_wrapper.hdf5_wrappers)):
        hdf5_path = load_data_result.multi_hdf5_wrapper.hdf5_wrappers[i].hdf5_path
        print(hdf5_path)
        print('all')
        compute_num_players_per_trajectory_one_hdf5_wrapper(loaded_model,
                                                            load_data_result.multi_hdf5_wrapper.hdf5_wrappers[i],
                                                            hdf5_path, False)
        print('train')
        compute_num_players_per_trajectory_one_hdf5_wrapper(loaded_model,
                                                            load_data_result.multi_hdf5_wrapper.train_hdf5_wrappers[i],
                                                            hdf5_path, True)
        print('test')
        compute_num_players_per_trajectory_one_hdf5_wrapper(loaded_model,
                                                            load_data_result.multi_hdf5_wrapper.test_hdf5_wrappers[i],
                                                            hdf5_path, False)


def verify_all_trajectories_in_first_wrapper_labeled(load_data_result: LoadDataResult):
    first_hdf5_wrapper = load_data_result.multi_hdf5_wrapper.hdf5_wrappers[0]
    round_ids = first_hdf5_wrapper.id_df[round_id_column].unique()
    labels = PushSaveRoundLabels(default_save_push_round_labels_path)
    for round_id in round_ids:
        if round_id not in labels.round_id_to_data:
            print(f'unlabled round id {round_id}')


if __name__ == "__main__":
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result)
    compute_num_players_per_trajectory(loaded_model, load_data_result)
    verify_all_trajectories_in_first_wrapper_labeled(load_data_result)
