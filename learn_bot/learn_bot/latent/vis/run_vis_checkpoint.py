import sys
from pathlib import Path

import pandas as pd
import torch

from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.latent_hdf5_dataset import LatentHDF5Dataset
from learn_bot.latent.place_area.filter import filter_region
from learn_bot.latent.place_area.pos_abs_delta_conversion import delta_pos_grid_num_cells, AABB
from learn_bot.latent.place_area.column_names import round_id_column, place_area_input_column_types, \
    delta_pos_output_column_types, test_success_col
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.latent.train import checkpoints_path, TrainResult, manual_latent_team_hdf5_data_path, \
    human_latent_team_hdf5_data_path, rollout_latent_team_hdf5_data_path, ColumnsToFlip
from learn_bot.libs.df_grouping import make_index_column, train_test_split_by_col_ids
from learn_bot.latent.vis.off_policy_inference import off_policy_inference
from learn_bot.latent.vis.vis import vis
from learn_bot.libs.vec import Vec3


def load_model_file(hdf5_data: HDF5Wrapper, model_file_name: str) -> TrainResult:
    cur_checkpoints_path = checkpoints_path
    if len(sys.argv) > 1:
        cur_checkpoints_path = cur_checkpoints_path / sys.argv[1]
    model_file = torch.load(cur_checkpoints_path / model_file_name)

    if model_file['diff_test_train']:
        train_test_split = train_test_split_by_col_ids(all_data_df, round_id_column, model_file['train_group_ids'])
        train_df = train_test_split.train_df.copy()
        make_index_column(train_df)
        test_df = train_test_split.test_df.copy()
        make_index_column(test_df)
    else:
        make_index_column(all_data_df)
        train_df = all_data_df
        test_df = all_data_df

    column_transformers = IOColumnTransformers(place_area_input_column_types, delta_pos_output_column_types, train_df)

    train_data = LatentHDF5Dataset(train_df, model_file['column_transformers'])
    test_data = LatentDataset(test_df, model_file['column_transformers'])

    model = TransformerNestedHiddenLatentModel(model_file['column_transformers'], 2 * max_enemies, delta_pos_grid_num_cells, 2, 4)
    model.load_state_dict(model_file['model_state_dict'])
    model.to(CUDA_DEVICE_STR)

    return TrainResult(train_data, test_data, train_df, test_df, column_transformers, model)


def load_model_file_for_rollout(all_data_df: pd.DataFrame, model_file_name: str) -> TrainResult:
    cur_checkpoints_path = checkpoints_path
    if len(sys.argv) > 1:
        cur_checkpoints_path = cur_checkpoints_path / sys.argv[1]
    model_file = torch.load(cur_checkpoints_path / model_file_name)

    make_index_column(all_data_df)

    all_data = LatentDataset(all_data_df, model_file['column_transformers'])

    column_transformers = IOColumnTransformers(place_area_input_column_types, delta_pos_output_column_types, all_data_df)

    model = TransformerNestedHiddenLatentModel(model_file['column_transformers'], 2 * max_enemies, delta_pos_grid_num_cells, 2, 4)
    model.load_state_dict(model_file['model_state_dict'])
    model.to(CUDA_DEVICE_STR)

    return TrainResult(all_data, all_data, all_data_df, all_data_df, column_transformers, model)


manual_data = False
rollout_data = False

if __name__ == "__main__":
    if manual_data:
        all_data_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=[i for i in range(20000)])
        #all_data_df = all_data_df[all_data_df['test name'] == b'LearnedGooseToCatScript']
    elif rollout_data:
        all_data_df = load_hdf5_to_pd(rollout_latent_team_hdf5_data_path)
    else:
        all_data_df = load_hdf5_to_pd(human_latent_team_hdf5_data_path)
    #all_data_df = all_data_df[all_data_df[test_success_col] == 1.]
    all_data_df = all_data_df.copy()

    #all_data_df = filter_region(all_data_df, AABB(Vec3(-580., 1740., 0.), Vec3(-280., 2088., 0.)), True, False,
    #                            [1, 2, 3, 4])

    #for flip_column in [ColumnsToFlip(" CT 1", " CT 2")]:
    #    flip_column.apply_flip(all_data_df)

    if not manual_data:
        load_result = load_model_file_for_rollout(all_data_df, "delta_pos_checkpoint.pt")
    else:
        load_result = load_model_file(all_data_df, "delta_pos_checkpoint.pt")

    pred_df = off_policy_inference(load_result.train_dataset, load_result.model, load_result.column_transformers)
    vis(load_result.train_df, pred_df)
