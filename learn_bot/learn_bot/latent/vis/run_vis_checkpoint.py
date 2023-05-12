import pandas as pd
import torch

from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.order.column_names import delta_pos_grid_num_cells
from learn_bot.latent.place_area.column_names import round_id_column, place_area_input_column_types, \
    delta_pos_output_column_types
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.latent.train import checkpoints_path, TrainResult, manual_latent_team_hdf5_data_path, \
    latent_team_hdf5_data_path
from learn_bot.libs.df_grouping import make_index_column, train_test_split_by_col
from learn_bot.latent.vis.off_policy_inference import off_policy_inference
from learn_bot.latent.vis.vis import vis


def load_model_file(all_data_df: pd.DataFrame, model_file_name: str) -> TrainResult:
    model_file = torch.load(checkpoints_path / model_file_name)

    if model_file['diff_test_train']:
        train_test_split = train_test_split_by_col(all_data_df, round_id_column)
        train_df = train_test_split.train_df.copy()
        make_index_column(train_df)
        test_df = train_test_split.test_df.copy()
        make_index_column(test_df)
    else:
        make_index_column(all_data_df)
        train_df = all_data_df
        test_df = all_data_df

    column_transformers = IOColumnTransformers(place_area_input_column_types, delta_pos_output_column_types, train_df)

    train_data = LatentDataset(train_df, model_file['column_transformers'])
    test_data = LatentDataset(test_df, model_file['column_transformers'])

    model = TransformerNestedHiddenLatentModel(model_file['column_transformers'], 2 * max_enemies, delta_pos_grid_num_cells)
    model.load_state_dict(model_file['model_state_dict'])
    model.to(CUDA_DEVICE_STR)

    return TrainResult(train_data, test_data, train_df, test_df, column_transformers, model)


manual_data = True

if __name__ == "__main__":
    if manual_data:
        all_data_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path)
    else:
        all_data_df = load_hdf5_to_pd(latent_team_hdf5_data_path)
        all_data_df = all_data_df[(all_data_df['valid'] == 1.) & (all_data_df['c4 status'] < 2)]
    all_data_df = all_data_df.copy()

    load_result = load_model_file(all_data_df, "delta_pos_checkpoint.pt")

    pred_df = off_policy_inference(load_result.test_dataset, load_result.model, load_result.column_transformers)
    vis(all_data_df, pred_df)
