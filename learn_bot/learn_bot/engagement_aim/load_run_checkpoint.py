import pandas as pd
import torch

from learn_bot.engagement_aim.column_names import target_o_float_columns, base_abs_x_pos_column, base_abs_y_pos_column, \
    input_column_types, output_column_types, all_time_column_types, engagement_id_column
from learn_bot.engagement_aim.dad import on_policy_inference
from learn_bot.engagement_aim.dataset import data_path, AimDataset, manual_data_path
from learn_bot.engagement_aim.io_transforms import FUTURE_TICKS, IOColumnTransformers, CUDA_DEVICE_STR
from learn_bot.engagement_aim.mlp_aim_model import MLPAimModel
from learn_bot.engagement_aim.train import checkpoints_path, TrainResult
from learn_bot.libs.df_grouping import train_test_split_by_col_ids, make_index_column
from learn_bot.libs.temporal_column_names import get_temporal_field_str
from learn_bot.engagement_aim.vis import vis
from dataclasses import dataclass



def load_model_file(all_data_df: pd.DataFrame, model_file_name: str) -> TrainResult:
    model_file = torch.load(checkpoints_path / model_file_name)
    train_test_split = train_test_split_by_col_ids(all_data_df, engagement_id_column, model_file['train_group_ids'])
    train_df = train_test_split.train_df.copy()
    make_index_column(train_df)
    test_df = train_test_split.test_df.copy()
    make_index_column(test_df)

    column_transformers = IOColumnTransformers(input_column_types, output_column_types, train_df)
    all_time_column_transformers = IOColumnTransformers(all_time_column_types, output_column_types, train_df)


    train_data = AimDataset(train_df, column_transformers, all_time_column_transformers)
    test_data = AimDataset(test_df, column_transformers, all_time_column_transformers)

    model = MLPAimModel(column_transformers)
    model.load_state_dict(model_file['model_state_dict'])
    model.to(CUDA_DEVICE_STR)

    return TrainResult(train_data, test_data, train_df, test_df, column_transformers,
                       all_time_column_transformers, model)


orig_dataset = False
train_dataset = False

if __name__ == "__main__":
    if orig_dataset:
        all_data_df = pd.read_csv(data_path)
    else:
        all_data_df = pd.read_csv(manual_data_path)

    all_data_df[target_o_float_columns[0]] = all_data_df[get_temporal_field_str(base_abs_x_pos_column, FUTURE_TICKS)]
    all_data_df[target_o_float_columns[1]] = all_data_df[get_temporal_field_str(base_abs_y_pos_column, FUTURE_TICKS)]

    #load_result = load_model_file(all_data_df, "model_off_2_scheduled_0_on_0_dad_0.pt")
    load_result = load_model_file(all_data_df, "model_off_5_scheduled_5_on_20_dad_1.pt")

    if orig_dataset:
        if train_dataset:
            dataset = load_result.train_dataset
            df = load_result.train_df
        else:
            dataset = load_result.test_dataset
            df = load_result.test_df
    else:
        df = all_data_df
        make_index_column(df)
        dataset = AimDataset(df, load_result.column_transformers, load_result.all_time_column_transformers)

    pred_df = on_policy_inference(dataset, df,
                                  load_result.model, load_result.column_transformers,
                                  True)
    vis.vis(df, pred_df)
