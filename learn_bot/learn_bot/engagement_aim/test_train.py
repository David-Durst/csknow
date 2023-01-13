import math
from typing import Dict

import torch.optim
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from dataset import *
from learn_bot.engagement_aim.mlp_aim_model import MLPAimModel
from learn_bot.engagement_aim.row_rollout import row_rollout, get_off_policy_blend_amount, get_on_policy_blend_amount, \
    get_scheduled_sampling_blend_amount
from learn_bot.engagement_aim.target_mlp_aim_model import TargetMLPAimModel
from learn_bot.engagement_aim.test_mlp_aim_model import TestMLPAimModel
from learn_bot.engagement_aim.train import checkpoints_path
from learn_bot.libs.accuracy_and_loss import compute_loss, compute_accuracy, finish_accuracy, CUDA_DEVICE_STR, \
    CPU_DEVICE_STR, AimLosses
from learn_bot.engagement_aim.lstm_aim_model import LSTMAimModel
from learn_bot.engagement_aim.output_plotting import plot_untransformed_and_transformed, ModelOutputRecording
from learn_bot.libs.df_grouping import train_test_split_by_col, make_index_column
from learn_bot.engagement_aim.dad import on_policy_inference, create_dad_dataset
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime

if __name__ == "__main__":
    all_data_df = pd.read_csv(manual_data_path)
    make_index_column(all_data_df)
    column_transformers = IOColumnTransformers(input_column_types, output_column_types, all_data_df)
    all_time_column_transformers = IOColumnTransformers(all_time_column_types, output_column_types, all_data_df)
    dataset = AimDataset(all_data_df, column_transformers, all_time_column_transformers)
    test_model = TestMLPAimModel(column_transformers)
    test_model.eval()
    script_test_model = torch.jit.trace(test_model.to(CPU_DEVICE_STR), dataset[0:1][0])
    output_test_model = test_model(dataset[1:2][0])
    output_test_script_model = script_test_model(dataset[1:2][0])
    print(torch.equal(output_test_model, output_test_script_model))

    model_file = torch.load(checkpoints_path / "model_off_5_scheduled_5_on_20_dad_1.pt")
    model = MLPAimModel(model_file['column_transformers'])
    model.load_state_dict(model_file['model_state_dict'])
    model.eval()
    script_model = torch.jit.trace(model.to(CPU_DEVICE_STR), dataset[0:1][0])
    output_model = model(dataset[1:2][0])
    output_script_model = script_model(dataset[1:2][0])
    print(torch.equal(output_model[0], output_script_model[0]))

    loaded_script_model = torch.jit.load(manual_data_path.parent / 'fixed_serialization_reduced_model.pt')
    output_loaded_script_model = loaded_script_model(dataset[1:2][0])
    print(torch.equal(output_model[0], output_loaded_script_model[0]))

    all_equal = True
    for p1, p2 in zip(script_model.state_dict().items(), loaded_script_model.state_dict().items()):
        if not torch.equal(p1[1], p2[1]):
            all_equal = False
            break
    print(all_equal)

    script_model_code_constants = script_model.code_with_constants
    loaded_script_model_code_constants = loaded_script_model.code_with_constants
    print(script_model_code_constants[0] == loaded_script_model_code_constants[0])
    #print(script_model_code_constants[0])

    script_constants = script_model_code_constants[1].const_mapping
    loaded_constants = loaded_script_model_code_constants[1].const_mapping
    print("start comparing const mapping")
    for k in script_constants.keys():
        print(f"{k}: {torch.equal(script_constants[k], loaded_constants[k])}")
    script_model.save(manual_data_path.parent / 'new_reduced_weighted_firing_script_model.pt')

    new_loaded_script_model = torch.jit.load(manual_data_path.parent / 'new_reduced_weighted_firing_script_model.pt')
    output_new_loaded_script_model = new_loaded_script_model(dataset[1:2][0])
    print(torch.equal(output_model[0], output_new_loaded_script_model[0]))

    loaded_script_model.save(manual_data_path.parent / 'resaved_reduced_weighted_firing_script_model.pt')
    reloaded_script_model = torch.jit.load(manual_data_path.parent / 'resaved_reduced_weighted_firing_script_model.pt')
    all_equal = True
    for p1, p2 in zip(loaded_script_model.state_dict().items(), reloaded_script_model.state_dict().items()):
        if not torch.equal(p1[1], p2[1]):
            all_equal = False
            break
    print(all_equal)

    reloaded_script_constants = reloaded_script_model.code_with_constants[1].const_mapping
    print("start comparing const mapping")
    for k in reloaded_script_constants.keys():
        print(f"{k}: {torch.equal(reloaded_script_constants[k], loaded_constants[k])}")
