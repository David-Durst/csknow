# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import math
from typing import Dict

import torch.optim
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from learn_bot.engagement_aim.dataset import *
from learn_bot.engagement_aim.mlp_aim_model import MLPAimModel
from learn_bot.engagement_aim.row_rollout import row_rollout, get_off_policy_blend_amount, get_on_policy_blend_amount, \
    get_scheduled_sampling_blend_amount
from learn_bot.engagement_aim.target_mlp_aim_model import TargetMLPAimModel
from learn_bot.libs.accuracy_and_loss import compute_loss, compute_accuracy, finish_accuracy, CUDA_DEVICE_STR, \
    CPU_DEVICE_STR, AimLosses
from learn_bot.engagement_aim.lstm_aim_model import LSTMAimModel
from learn_bot.engagement_aim.output_plotting import ModelOutputRecording
from learn_bot.libs.plot_features import plot_untransformed_and_transformed
from learn_bot.libs.df_grouping import train_test_split_by_col, make_index_column
from learn_bot.engagement_aim.dad import on_policy_inference, create_dad_dataset
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime

from learn_bot.engagement_aim.vis import vis
from learn_bot.libs.compare_models import *
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

checkpoints_path = Path(__file__).parent / 'checkpoints'

now = datetime.now()
runs_path = Path(__file__).parent / 'runs' / now.strftime("%m_%d_%Y__%H_%M_%S")


@dataclass(frozen=True)
class TrainResult:
    train_dataset: AimDataset
    test_dataset: AimDataset
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    column_transformers: IOColumnTransformers
    all_time_column_transformers: IOColumnTransformers
    model: nn.Module


def train(all_data_df: pd.DataFrame, dad_iters=4, num_off_policy_epochs=5, num_scheduled_sampling_epochs=5,
          num_on_policy_epochs=5, save=True, diff_train_test=True) -> TrainResult:
    # all_data_df = all_data_df[all_data_df['num shots fired'] > 0]

    if diff_train_test:
        train_test_split = train_test_split_by_col(all_data_df, engagement_id_column)
        train_df = train_test_split.train_df.copy()
        train_group_ids = train_test_split.train_group_ids
        make_index_column(train_df)
        test_df = train_test_split.test_df.copy()
        make_index_column(test_df)
    else:
        make_index_column(all_data_df)
        train_df = all_data_df
        train_group_ids = list(all_data_df.loc[:, engagement_id_column].unique())
        test_df = all_data_df


    # transform input and output
    column_transformers = IOColumnTransformers(input_column_types, output_column_types, train_df)
    all_time_column_transformers = IOColumnTransformers(all_time_column_types, output_column_types, train_df)

    # plot data set with and without transformers
    plot_untransformed_and_transformed('train and test labels', all_data_df,
                                       temporal_io_float_standard_column_names.present_columns,
                                       temporal_io_cat_column_names.present_columns + static_input_categorical_columns)

    # Get cpu or gpu device for training.
    device: str = CUDA_DEVICE_STR if torch.cuda.is_available() else CPU_DEVICE_STR
    # device = CPU_DEVICE_STR
    print(f"Using {device} device")

    # Define model
    embedding_dim = 5
    model = MLPAimModel(column_transformers).to(device)
    # model = LSTMAimModel(column_transformers, len(temporal_io_float_column_names.input_columns), len(non_temporal_float_columns)).to(device)

    print(model)
    params = list(model.parameters())
    print("params by layer")
    for param_layer in params:
        print(param_layer.shape)

    # define losses
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    #scheduler = ReduceLROnPlateau(optimizer, 'min')

    # train and test the model
    first_row: torch.Tensor
    #model_output_recording: ModelOutputRecording = ModelOutputRecording(model)
    time_weights_list = [1]
    for _ in range(FUTURE_TICKS):
        time_weights_list.append(0.8 * time_weights_list[-1])
    time_weights = torch.tensor([time_weights_list])

    def train_or_test_SL_epoch(dataloader, model, optimizer, first_epoch, last_epoch, blend_amount, include_cat_cols,
                               train=True):
        nonlocal first_row#, model_output_recording
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        if train:
            model.train()
        else:
            model.eval()
        cumulative_loss = AimLosses()
        accuracy = {}
        # bar = Bar('Processing', max=size)
        for name in column_transformers.output_types.column_names():
            accuracy[name] = 0
        with tqdm(total=len(dataloader), disable=False) as pbar:
            for batch, (X, Y, targets, attacking, all_time_X, idxs) in enumerate(dataloader):
                if batch == 0 and first_epoch and train:
                    first_row = X[0:1, :]
                X, Y, all_time_X = X.to(device), Y.to(device), all_time_X.to(device)
                #transformed_X = column_transformers.transform_columns(True, X, X)
                transformed_Y = column_transformers.transform_columns(False, Y, X)
                #untransformed_Y = column_transformers.untransform_columns(False, transformed_Y, X)
                #transformed_targets = column_transformers.transform_columns(False, targets, X)
                #input_name_ranges_dict = column_transformers.get_name_ranges_dict(True, True)
                transformed_last_input_angles = None#\
                #    transformed_X[:, [input_name_ranges_dict[get_temporal_field_str(base_abs_x_pos_column, -1)].start,
                #                      input_name_ranges_dict[get_temporal_field_str(base_abs_y_pos_column, -1)].start]]
                # XR = torch.randn_like(X, device=device)
                # XR[:,0] = X[:,0]
                # YZ = torch.zeros_like(Y) + 0.1

                # Compute prediction error
                #pred2 = model(X)
                pred = row_rollout(model, X, all_time_X, transformed_Y, Y, all_time_column_transformers,
                                    column_transformers, blend_amount)
                batch_loss = compute_loss(X, pred, transformed_Y, Y, targets, attacking,
                                          transformed_last_input_angles, time_weights, column_transformers,
                                          include_cat_cols)
                cumulative_loss += batch_loss

                # Backpropagation
                if train:
                    optimizer.zero_grad()
                    batch_loss.get_total_loss().backward()
                    optimizer.step()

                if False and train and batch % 100 == 0:
                    loss, current = batch_loss.get_total_loss().item(), batch * len(X)
                    print('pred')
                    print(pred[0:2])
                    print('y')
                    print(Y[0:2])
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                compute_accuracy(pred, Y, accuracy, column_transformers)
                #if last_epoch:
                #    model_output_recording.record_output(pred, Y, transformed_Y, train)
                pbar.update(1)

        cumulative_loss /= num_batches
        for name in column_transformers.output_types.column_names():
            accuracy[name] /= size
        accuracy_string = finish_accuracy(accuracy, column_transformers)
        train_test_str = "Train" if train else "Test"
        print(f"Epoch {train_test_str} Accuracy: {accuracy_string}, Transformed Avg Loss: {cumulative_loss.get_total_loss().item():>8f}")
        return cumulative_loss, accuracy

    def save_model(best: bool):
        shared_str = f"off_{num_off_policy_epochs}_scheduled_" \
                     f"{num_scheduled_sampling_epochs}_on_{num_on_policy_epochs}_dad_{dad_iters}"
        if best:
            name = f"best_model_{shared_str}.pt"
        else:
            name = f"model_{shared_str}.pt"
        torch.save({
            'train_group_ids': train_group_ids,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'column_transformers': column_transformers,
            'all_time_column_transformers': all_time_column_transformers
        }, checkpoints_path / name)

    writer = SummaryWriter(runs_path)
    def save_tensorboard(train_loss: AimLosses, test_loss: AimLosses, train_accuracy: Dict, test_accuracy: Dict,
                         total_epoch_num):
        train_loss.add_scalars(writer, 'train', total_epoch_num)
        test_loss.add_scalars(writer, 'test', total_epoch_num)
        for name, acc in train_accuracy.items():
            writer.add_scalar('train/acc/' + name, acc, total_epoch_num)
        for name, acc in test_accuracy.items():
            writer.add_scalar('test/acc/' + name, acc, total_epoch_num)

    best_result = None
    def train_and_test_SL(model, train_dataloader, test_dataloader, num_epochs, start_overall_epoch, blend_amount_fn,
                          first_epoch_set, last_epoch_set, include_cat_cols):
        nonlocal optimizer, best_result
        for epoch_num in range(num_epochs):
            overall_epoch_num = start_overall_epoch + epoch_num
            print(f"\nEpoch {overall_epoch_num}, Blend {blend_amount_fn(epoch_num, num_epochs)}\n"
                  f"-------------------------------")
            #if epoch_num % 100 == 1000:
                # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            first_epoch = first_epoch_set and epoch_num == 0
            last_epoch = last_epoch_set and epoch_num == num_epochs - 1
            train_loss, train_accuracy = train_or_test_SL_epoch(train_dataloader, model, optimizer,
                                                                first_epoch, last_epoch,
                                                                blend_amount_fn(epoch_num, num_epochs),
                                                                include_cat_cols, True)
            with torch.no_grad():
                test_loss, test_accuracy = train_or_test_SL_epoch(test_dataloader, model, None,
                                                                  first_epoch, last_epoch,
                                                                  blend_amount_fn(epoch_num, num_epochs),
                                                                  include_cat_cols, False)
            if (epoch_num > 0 and epoch_num % 20 == 0) or epoch_num == num_epochs - 1:
                save_model(False)
            if best_result is None or test_loss.get_total_loss() < best_result.get_total_loss():
                best_result = test_loss
                save_model(True)
            save_tensorboard(train_loss, test_loss, train_accuracy, test_accuracy, overall_epoch_num)
            #scheduler.step(train_loss)

    total_train_df = train_df
    train_data = AimDataset(train_df, column_transformers, all_time_column_transformers)
    test_data = AimDataset(test_df, column_transformers, all_time_column_transformers)
    num_overall_epochs = 0
    for dad_num in range(dad_iters + 1):
        if dad_num < dad_iters:
            print(f"DaD Iter {dad_num}\n-------------------------------")
        else:
            print(f"Final, Non-DaD Iter\n-------------------------------")
        # step 1: train model
        # create data sets for pytorch
        total_train_data = AimDataset(total_train_df, column_transformers, all_time_column_transformers)

        batch_size = min([128, len(total_train_data), len(test_data)])

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        total_train_dataloader = DataLoader(total_train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        print(f"num train examples: {len(total_train_data)}")
        print(f"num test examples: {len(test_data)}")

        if dad_num == 0:
            for X, Y, target, attacking, _, _ in total_train_dataloader:
                print(f"Train shape of X: {X.shape} {X.dtype}")
                print(f"Train shape of Y: {Y.shape} {Y.dtype}")
                print(f"Train shape of target: {target.shape} {target.dtype}")
                print(f"Train shape of attacking: {attacking.shape} {attacking.dtype}")
                break

            for X, Y, target, attacking, _, _ in test_dataloader:
                print(f"Test shape of X: {X.shape} {X.dtype}")
                print(f"Test shape of Y: {Y.shape} {Y.dtype}")
                print(f"Test shape of target: {target.shape} {target.dtype}")
                print(f"Test shape of attacking: {attacking.shape} {attacking.dtype}")
                break

        epochs_per_dad_iter = num_off_policy_epochs + num_scheduled_sampling_epochs + num_on_policy_epochs

        if num_off_policy_epochs > 0:
            train_and_test_SL(model, total_train_dataloader, test_dataloader, num_off_policy_epochs, num_overall_epochs,
                              get_off_policy_blend_amount,
                              dad_num == 0,
                              dad_num == dad_iters and epochs_per_dad_iter == num_off_policy_epochs, True)
            num_overall_epochs += num_off_policy_epochs

        if num_scheduled_sampling_epochs > 0:
            train_and_test_SL(model, total_train_dataloader, test_dataloader, num_scheduled_sampling_epochs, num_overall_epochs,
                              get_scheduled_sampling_blend_amount,
                              dad_num == 0 and num_off_policy_epochs == 0,
                              dad_num == dad_iters and
                              epochs_per_dad_iter == num_off_policy_epochs + num_scheduled_sampling_epochs, True)
            num_overall_epochs += num_scheduled_sampling_epochs

        if num_on_policy_epochs > 0:
            train_and_test_SL(model, total_train_dataloader, test_dataloader, num_on_policy_epochs, num_overall_epochs,
                              get_on_policy_blend_amount,
                              dad_num == 0 and num_off_policy_epochs == 0 and num_scheduled_sampling_epochs == 0,
                              dad_num == dad_iters, True)
            num_overall_epochs += num_on_policy_epochs

        if False and dad_num == dad_iters and num_off_policy_epochs > 0:
            train_and_test_SL(model, train_dataloader, test_dataloader, num_off_policy_epochs, num_overall_epochs,
                              get_off_policy_blend_amount,
                              dad_num == 0, True, True)

        if dad_num < dad_iters:
            # step 2: inference and result collection
            pred_df = on_policy_inference(train_data, train_df, model, column_transformers)
            # model.to(CUDA_DEVICE_STR)

            # step 3: create new training data set
            dad_df = create_dad_dataset(pred_df, train_df)
            total_train_df = pd.concat([total_train_df, dad_df], ignore_index=True)

    #model_output_recording.plot(column_transformers,
    #                            output_column_types.float_standard_cols + output_column_types.delta_float_column_names() +
    #                            output_column_types.float_180_angle_cols + output_column_types.delta_180_angle_column_names() +
    #                            output_column_types.float_90_angle_cols + output_column_types.delta_90_angle_column_names(),
    #                            output_column_types.categorical_cols)

    if save:
        script_model = torch.jit.trace(model.to(CPU_DEVICE_STR), first_row)
        script_model.save(Path(__file__).parent / '..' / '..' / 'models' / 'engagement_aim_model' / 'script_model.pt')
        model.to(device)

    return TrainResult(train_data, test_data, train_df, test_df, column_transformers,
                       all_time_column_transformers, model)


if __name__ == "__main__":
    all_data_df = load_hdf5_to_pd(latent_hdf5_data_path)
    #all_data_df = load_hdf5_to_pd(hdf5_data_path)
    #all_data_df = pd.read_csv(data_path)
    all_data_df[target_o_float_columns[0]] = \
        all_data_df[get_temporal_field_str(base_changed_offset_coordinates.attacker_x_view_angle, FUTURE_TICKS)]
    all_data_df[target_o_float_columns[1]] = \
        all_data_df[get_temporal_field_str(base_changed_offset_coordinates.attacker_y_view_angle, FUTURE_TICKS)]
    #all_data_df = all_data_df[(all_data_df[weapon_type_col] == 3) & (all_data_df[cur_victim_visible_yet_column] == 1.)]
    train_result = train(all_data_df, dad_iters=1, num_off_policy_epochs=5, num_scheduled_sampling_epochs=5,
                         num_on_policy_epochs=20)
    #engagement_ids = list(train_result.test_df[engagement_id_column].unique())
    #engagement_ids = engagement_ids[:30]
    #limited_test_df = train_result.test_df[train_result.test_df[engagement_id_column].isin(engagement_ids)]
    #limited_test_dataset = AimDataset(limited_test_df, train_result.column_transformers)
    #pred_df = on_policy_inference(train_result.test_dataset, train_result.test_df,
    #                              train_result.model, train_result.column_transformers,
    #                              True)
    #vis.vis(train_result.test_df, pred_df)
