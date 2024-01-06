# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import os
import random
import time
from enum import Enum
from typing import Dict
import sys

import torch.optim
from torch import autocast
from torch.profiler import profile, ProfilerActivity, schedule
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from learn_bot.latent.analyze.comparison_column_names import small_human_good_rounds, \
    all_human_28_second_filter_good_rounds, all_human_vs_small_human_similarity_hdf5_data_path, \
    all_human_vs_human_28_similarity_hdf5_data_path
from learn_bot.latent.latent_subset_hdf5_dataset import *
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.hyperparameter_options import HyperparameterOptions
from learn_bot.latent.latent_hdf5_dataset import MultipleLatentHDF5Dataset
from learn_bot.latent.place_area.load_data import human_latent_team_hdf5_data_path, manual_latent_team_hdf5_data_path, \
    LoadDataResult, LoadDataOptions
from learn_bot.latent.place_area.column_names import place_area_input_column_types, radial_vel_output_column_types, \
    test_success_col
from learn_bot.latent.place_area.simulation.rollout_simulator import rollout_simulate
from learn_bot.latent.profiling import profile_latent_model
from learn_bot.latent.train_paths import checkpoints_path, runs_path, train_test_split_file_name, \
    default_selected_retake_rounds_path
from learn_bot.latent.transformer_10_17 import TransformerNestedHiddenLatentModel_10_17
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel, PlayerMaskType, \
    OutputMaskType, ControlType
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper
from learn_bot.libs.io_transforms import CUDA_DEVICE_STR
from learn_bot.latent.accuracy_and_loss import compute_loss, compute_accuracy_and_delta_diff, \
    finish_accuracy_and_delta_diff, \
    CPU_DEVICE_STR, LatentLosses, duplicated_name_str, compute_output_mask, TotalMaskStatistics, \
    compute_total_mask_statistics
from learn_bot.libs.multi_hdf5_wrapper import MultiHDF5Wrapper
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

time_model = False

class TrainType(Enum):
    DeltaPos = 1


default_hyperparameter_options = HyperparameterOptions()
hyperparameter_option_range = [HyperparameterOptions(num_input_time_steps=1, control_type=ControlType.SimilarityControl),
                               HyperparameterOptions(num_input_time_steps=3, control_type=ControlType.SimilarityControl),
                               HyperparameterOptions(num_input_time_steps=1, control_type=ControlType.TimeControl),
                               HyperparameterOptions(num_input_time_steps=1, control_type=ControlType.NoControl),
                               HyperparameterOptions(num_input_time_steps=1, control_type=ControlType.SimilarityControl,
                                                     mask_partial_info=True),
                               HyperparameterOptions(num_input_time_steps=3, control_type=ControlType.SimilarityControl,
                                                     mask_partial_info=True),
                               HyperparameterOptions(num_input_time_steps=1, control_type=ControlType.NoControl),
                               HyperparameterOptions(num_input_time_steps=1, internal_width=128, layers=2, control_type=ControlType.TimeControl),
                               HyperparameterOptions(num_input_time_steps=1, layers=8, heads=8, bc_epochs=40, learning_rate=1e-5, control_type=ControlType.TimeControl),
                               HyperparameterOptions(num_input_time_steps=1, layers=16, internal_width=256, heads=8, bc_epochs=40, learning_rate=1e-5, control_type=ControlType.TimeControl),
                               HyperparameterOptions(num_input_time_steps=1, layers=8, bc_epochs=40, learning_rate=1e-5, control_type=ControlType.TimeControl),
                               HyperparameterOptions(num_input_time_steps=1, layers=8, bc_epochs=40, learning_rate=5e-6, control_type=ControlType.TimeControl),
                               HyperparameterOptions(num_input_time_steps=1, layers=8, bc_epochs=40, learning_rate=1e-6, control_type=ControlType.TimeControl),
                               HyperparameterOptions(num_input_time_steps=1, control_type=ControlType.SimilarityControl),
                               HyperparameterOptions(num_input_time_steps=5, bc_epochs=60, internal_width=1024, layers=24, learning_rate=1e-6, control_type=ControlType.TimeControl),
                               HyperparameterOptions(num_input_time_steps=5, bc_epochs=60, internal_width=1024, layers=24, control_type=ControlType.SimilarityControl),
                               HyperparameterOptions(num_input_time_steps=1, control_type=ControlType.SimilarityControl,
                                                     layers=20, internal_width=512),
                               HyperparameterOptions(num_input_time_steps=1,
                                                     player_mask_type=PlayerMaskType.EveryoneMask),
                               HyperparameterOptions(num_input_time_steps=5,
                                                     output_mask=OutputMaskType.NoEngagementMask),
                               # pointless to have temporal mask with only one input time step
                               #HyperparameterOptions(num_input_time_steps=1,
                               #                      player_mask_type=PlayerMaskType.TeammateTemporalOnlyMask),
                               #HyperparameterOptions(num_input_time_steps=1,
                               #                      player_mask_type=PlayerMaskType.EveryoneTemporalOnlyMask),
                               HyperparameterOptions(num_input_time_steps=3, weight_not_move_loss=20.,
                                                     drop_history_probability=0.7),
                               HyperparameterOptions(num_input_time_steps=3, full_rollout_epochs=10),
                               HyperparameterOptions(num_input_time_steps=5, bc_epochs=40),
                               HyperparameterOptions(num_input_time_steps=25, bc_epochs=40),
                               HyperparameterOptions(layers=4, heads=8),
                               HyperparameterOptions(layers=8, heads=8),
                               HyperparameterOptions(weight_decay=0.1),
                               HyperparameterOptions(weight_decay=1e-3),
                               HyperparameterOptions(batch_size=256),
                               HyperparameterOptions(bc_epochs=1024),
                               HyperparameterOptions(bc_epochs=2048),
                               HyperparameterOptions(learning_rate=1e-4),
                               HyperparameterOptions(learning_rate=1e-5),
                               HyperparameterOptions(learning_rate=1e-6),
                               HyperparameterOptions(learning_rate=1e-7),
                               HyperparameterOptions(weight_decay=0.01),
                               HyperparameterOptions(weight_decay=1e-4)]
#hyperparameter_option_range = [HyperparameterOptions(learning_rate=1e-5),
#                               HyperparameterOptions(num_epochs=3000, learning_rate=1e-6),
#                               HyperparameterOptions(weight_decay=0.1),
#                               HyperparameterOptions(layers=4, heads=8)]
                               #HyperparameterOptions(noise_var=1.)]
#hyperparameter_option_range = [HyperparameterOptions(),
#                               HyperparameterOptions(learning_rate=4e-4),
#                               HyperparameterOptions(learning_rate=4e-6),
#                               HyperparameterOptions(weight_decay=0.1),
#                               HyperparameterOptions(weight_decay=0.2),
#                               HyperparameterOptions(layers=4, heads=8)]


@dataclass
class ColumnsToFlip:
    col1_template: str
    col2_template: str

    def apply_flip(self, df: pd.DataFrame):
        col1_columns = [col for col in df.columns if self.col1_template in col]
        col2_columns = [col for col in df.columns if self.col2_template in col]
        cols_to_swap_list = zip(col1_columns, col2_columns)
        cols_rename_map = {}
        for cols_to_swap in cols_to_swap_list:
            cols_rename_map[cols_to_swap[0]] = cols_to_swap[1]
            cols_rename_map[cols_to_swap[1]] = cols_to_swap[0]
        df.rename(columns=cols_rename_map, inplace=True)

    def check_flip(self, old_df: pd.DataFrame, new_df: pd.DataFrame):
        col1_columns = [col for col in old_df.columns if self.col1_template in col]
        col2_columns = [col for col in old_df.columns if self.col2_template in col]
        #print("flipped columns " + ",".join(col2_columns))
        for (col1, col2) in zip(col1_columns, col2_columns):
            assert (old_df[col1] == new_df[col2]).all()
            assert (old_df[col2] == new_df[col1]).all()

    def duplicate(self, df: pd.DataFrame):
        col1_columns = [col for col in df.columns if self.col1_template in col]
        col2_columns = [col for col in df.columns if self.col2_template in col]
        for (col1, col2) in zip(col1_columns, col2_columns):
            df[col2] = df[col1]


total_epochs = 0

@dataclass
class TrainCheckpointPaths:
    last_not_best_path: Path
    best_path: Path


def train(train_type: TrainType, multi_hdf5_wrapper: MultiHDF5Wrapper,
          hyperparameter_options: HyperparameterOptions = default_hyperparameter_options,
          diff_train_test=True, flip_columns: List[ColumnsToFlip] = [], load_model_path: Optional[Path] = None,
          enable_training: bool = True) -> TrainCheckpointPaths:
    train_checkpoint_paths: TrainCheckpointPaths = TrainCheckpointPaths(Path("INVALID"), Path("INVALID"))

    #if len(flip_columns) > 0:
    #    io_column_transform_df = all_data_df.copy(deep=True)
    #    for flip_column in flip_columns:
    #        flip_column.apply_flip(test_df)
    #        flip_column.check_flip(train_df, test_df)
    #        flip_column.duplicate(io_column_transform_df)
    #else:
    #    io_column_transform_df = all_data_df

    # Get cpu or gpu device for training.
    device: str = CUDA_DEVICE_STR if torch.cuda.is_available() else CPU_DEVICE_STR
    # device = CPU_DEVICE_STR
    print(f"Using {device} device")

    # Define model
    if train_type == TrainType.DeltaPos:
        column_transformers = IOColumnTransformers(place_area_input_column_types, radial_vel_output_column_types,
                                                   multi_hdf5_wrapper.train_hdf5_wrappers[0].sample_df)
        # plus 1 on future ticks to include present tick
        model = TransformerNestedHiddenLatentModel(column_transformers, hyperparameter_options.internal_width,
                                                   2 * max_enemies, hyperparameter_options.num_input_time_steps,
                                                   hyperparameter_options.layers, hyperparameter_options.heads,
                                                   hyperparameter_options.control_type,
                                                   hyperparameter_options.player_mask_type,
                                                   hyperparameter_options.mask_partial_info)
        if load_model_path:
            model_file = torch.load(load_model_path)
            model.load_state_dict(model_file['model_state_dict'])
        model = model.to(device)
        input_column_types = place_area_input_column_types
        output_column_types = radial_vel_output_column_types
    else:
        raise Exception("invalid train type")

    # plot data set with and without transformers
    #plot_untransformed_and_transformed(plot_path, 'train and test labels', all_data_df,
    #                                   input_column_types.float_standard_cols + output_column_types.categorical_distribution_cols_flattened,
    #                                   input_column_types.categorical_cols + output_column_types.categorical_cols)

    print(model)
    params = list(model.parameters())
    print("params by layer")
    for param_layer in params:
        print(param_layer.shape)

    run_checkpoints_path = hyperparameter_options.get_checkpoints_path()
    run_checkpoints_path.mkdir(parents=True, exist_ok=True)
    train_checkpoint_paths.best_path = run_checkpoints_path

    # define losses
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameter_options.learning_rate,
                                 weight_decay=hyperparameter_options.weight_decay)
    if load_model_path is not None:
        optimizer.load_state_dict(model_file['optimizer_state_dict'])
    scaler = torch.cuda.amp.GradScaler()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # train and test the model
    # first row used to provide input during serialization
    first_row: torch.Tensor = None
    first_row_similarity: torch.Tensor = None
    temperature_gpu = torch.Tensor([1.]).to(CUDA_DEVICE_STR)
    temperature_cpu = torch.Tensor([1.])

    def train_or_test_SL_epoch(dataloader, model, optimizer, scaler, train=True, profiler=None):
        nonlocal first_row, first_row_similarity
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        if train:
            model.train()
        else:
            model.eval()
        cumulative_loss = LatentLosses()
        accuracy = {}
        delta_diff_xy = {}
        delta_diff_xyz = {}
        valids_per_accuracy_column = {}
        #losses = []
        # bar = Bar('Processing', max=size)
        batch_num = 0
        start_epoch_time = time.perf_counter()
        dataloader.dataset.rollout_steps = hyperparameter_options.get_rollout_steps(total_epochs)
        total_mask_statistics = TotalMaskStatistics()
        with tqdm(total=len(dataloader), disable=False) as pbar:
            for batch, (X, Y, similarity, duplicated_last, indices) in enumerate(dataloader):
                batch_num += 1
                #if batch_num > 24:
                #    break
                if first_row is None:
                    if hyperparameter_options.get_rollout_steps(total_epochs) == 1:
                        first_row = X[0:1, :].float()
                        first_row_similarity = similarity[0:1, :].float()
                    else:
                        first_row = X[0:1, 0, :].float()
                        first_row_similarity = similarity[0:1, 0, :].float()
                X, Y, duplicated_last = X.to(device), Y.to(device), duplicated_last.to(device)
                Y = Y.float()
                similarity = similarity.to(device).float()
                # XR = torch.randn_like(X, device=device)
                # XR[:,0] = X[:,0]
                # YZ = torch.zeros_like(Y) + 0.1

                if time_model:
                    if train_type == TrainType.DeltaPos:
                        model_path = run_checkpoints_path / 'delta_pos_script_model.pt'
                    profile_latent_model(model_path, batch_size, X)

                # Compute prediction error
                if train:
                    model.noise_var = hyperparameter_options.noise_var
                    optimizer.zero_grad()
                X_flattened_orig = None
                X_flattened_rollout = None
                with autocast(device, enabled=True):
                    if hyperparameter_options.drop_history_probability is not None and \
                            random.random() < hyperparameter_options.drop_history_probability:
                        model.drop_history = True

                    if hyperparameter_options.get_rollout_steps(total_epochs) == 1:
                        pred = model(X, similarity, temperature_gpu)
                        pred_flattened = pred
                        Y_flattened = Y
                        duplicated_last_flattened = duplicated_last
                        output_mask = compute_output_mask(model, X, hyperparameter_options.output_mask)
                    else:
                        rollout_batch_result = \
                            rollout_simulate(X, Y, similarity, duplicated_last, indices, model,
                                             hyperparameter_options.percent_rollout_steps_predicted(total_epochs))
                        X_flattened_orig = rollout_batch_result.X_flattened_orig
                        X_flattened_rollout = rollout_batch_result.X_flattened_rollout
                        pred_flattened = rollout_batch_result.model_pred_flattened
                        Y_flattened = rollout_batch_result.Y_flattened
                        duplicated_last_flattened = rollout_batch_result.duplicated_last_flattened
                        output_mask = compute_output_mask(model, X_flattened_orig, hyperparameter_options.output_mask)

                    model.noise_var = -1.
                    model.drop_history = False
                    if torch.isnan(X).any():
                        print('bad X')
                        sys.exit(0)
                    if torch.isnan(Y_flattened).any():
                        print('bad Y')
                        sys.exit(0)
                    if torch.isnan(pred_flattened[0]).any():
                        print(X)
                        print(pred_flattened[0, 0])
                        #print(torch.isnan(pred_flattened[0,0]).nonzero())
                        #bad_batch_row = torch.isnan(pred_flattened[0,0]).nonzero()[0,0].item()
                        #print(indices[bad_batch_row])
                        #print('bad pred')
                        sys.exit(0)
                    compute_total_mask_statistics(Y, model.num_players, output_mask, total_mask_statistics)
                    batch_loss = compute_loss(model, pred_flattened, Y_flattened, X_flattened_orig, X_flattened_rollout,
                                              duplicated_last_flattened, model.num_players,
                                              output_mask,
                                              hyperparameter_options.weight_not_move_loss)
                    # uncomment here and below causes memory issues
                    cumulative_loss += batch_loss
                    #losses.append(batch_loss.total_loss.tolist()[0])

                # Backpropagation
                if train:
                    scaler.scale(batch_loss.total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                compute_accuracy_and_delta_diff(pred_flattened, Y_flattened, duplicated_last_flattened,
                                                accuracy, delta_diff_xy, delta_diff_xyz,
                                                valids_per_accuracy_column, model.num_players, column_transformers,
                                                model.stature_to_speed_gpu, output_mask)
                pbar.update(1)
                if profiler is not None:
                    profiler.step()

        end_epoch_time = time.perf_counter()
        cumulative_loss /= len(dataloader)
        for name in column_transformers.output_types.column_names():
            if name in valids_per_accuracy_column and valids_per_accuracy_column[name] > 0:
                # if nothing in the category, just make sure it isn't 0
                # this is used for repeated columns
                valids_per_accuracy_cur_column = valids_per_accuracy_column[name].item()
                if valids_per_accuracy_cur_column == 0.:
                    valids_per_accuracy_cur_column = 1.
                accuracy[name] = accuracy[name].item() / valids_per_accuracy_cur_column
                delta_diff_xy[name] = delta_diff_xy[name].item() / valids_per_accuracy_cur_column
                delta_diff_xyz[name] = delta_diff_xyz[name].item() / valids_per_accuracy_cur_column
                duplicated_valids_per_accuracy_cur_column = valids_per_accuracy_column[name + duplicated_name_str].item()
                if duplicated_valids_per_accuracy_cur_column == 0.:
                    duplicated_valids_per_accuracy_cur_column = 1.
                accuracy[name + duplicated_name_str] = \
                    accuracy[name + duplicated_name_str].item() / duplicated_valids_per_accuracy_cur_column
                delta_diff_xy[name + duplicated_name_str] = \
                    delta_diff_xy[name + duplicated_name_str].item() / duplicated_valids_per_accuracy_cur_column
                delta_diff_xyz[name + duplicated_name_str] = \
                    delta_diff_xyz[name + duplicated_name_str].item() / duplicated_valids_per_accuracy_cur_column
        accuracy_string = finish_accuracy_and_delta_diff(accuracy, delta_diff_xy, delta_diff_xyz,
                                                         valids_per_accuracy_column, column_transformers)
        train_test_str = "Train" if train else "Test"
        print(f"Epoch {train_test_str} Accuracy: {accuracy_string}, Transformed Avg Loss: {cumulative_loss.total_accumulator:>8f}")
        print(f"Percent Included By Mask: {total_mask_statistics.num_player_points_included_by_mask / float(total_mask_statistics.num_player_points):.4f}")
        print(f"Batch Time {(end_epoch_time - start_epoch_time) / batch_num: 0.4f} s")
        return cumulative_loss, accuracy, delta_diff_xy, delta_diff_xyz

    def save_model(not_best: bool, iter: int):
        nonlocal train_type, train_checkpoint_paths
        save_path = run_checkpoints_path
        if not_best:
            save_path = run_checkpoints_path / 'not_best' / str(iter)
            train_checkpoint_paths.last_not_best_path = save_path
            os.makedirs(save_path, exist_ok=True)
        model_path = save_path / 'delta_pos_checkpoint.pt'
        torch.save({
            'train_test_splits': multi_hdf5_wrapper.train_test_splits,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'column_transformers': column_transformers,
            'diff_test_train': diff_train_test,
            'hyperparameter_options': hyperparameter_options,
            'total_epochs': total_epochs
        }, model_path)
        with torch.no_grad():
            model.eval()
            script_model = torch.jit.trace(model.to(CPU_DEVICE_STR), (first_row, first_row_similarity, temperature_cpu))
            #tmp_model = SimplifiedTransformerNestedHiddenLatentModel().to(device)
            #tmp_model.to(CPU_DEVICE_STR)
            #tmp_model.eval()
            #torch.jit.trace(tmp_model, torch.ones([64, 10, 512]))
            #  torch.jit.script(tmp_model)
            if train_type == TrainType.DeltaPos:
                script_model.save(save_path / 'delta_pos_script_model.pt')
                with open(save_path / 'delta_pos_test_round_ids.csv', 'w+') as f:
                    for k, v in multi_hdf5_wrapper.test_group_ids.items():
                        f.write(f'''{str(k)} : {','.join(map(str, v))}''')
            model.to(device)

    cur_runs_path = runs_path / str(hyperparameter_options)
    writer = SummaryWriter(cur_runs_path)
    def save_tensorboard(train_loss: LatentLosses, test_loss: LatentLosses, train_accuracy: Dict, test_accuracy: Dict,
                         train_delta_diff_xy: Dict, test_delta_diff_xy: Dict,
                         train_delta_diff_xyz: Dict, test_delta_diff_xyz: Dict, epoch_num):
        train_loss.add_scalars(writer, 'train', epoch_num)
        test_loss.add_scalars(writer, 'test', epoch_num)
        for name, acc in train_accuracy.items():
            writer.add_scalar('train/acc/' + name, acc, epoch_num)
        for name, acc in test_accuracy.items():
            writer.add_scalar('test/acc/' + name, acc, epoch_num)
        for name, delta in train_delta_diff_xy.items():
            writer.add_scalar('train/delta_xy/' + name, delta, epoch_num)
        for name, delta in test_delta_diff_xy.items():
            writer.add_scalar('test/delta_xy/' + name, delta, epoch_num)
        for name, delta_z in train_delta_diff_xyz.items():
            writer.add_scalar('train/delta_xyz/' + name, delta_z, epoch_num)
        for name, delta_z in test_delta_diff_xyz.items():
            writer.add_scalar('test/delta_xyz/' + name, delta_z, epoch_num)

    min_test_loss = float("inf")
    def train_and_test_SL(model, train_dataloader, test_dataloader, num_epochs):
        global total_epochs
        nonlocal optimizer, scaler, min_test_loss
        for _ in range(num_epochs):
            print(f"\nEpoch {total_epochs}\n" + f"-------------------------------")
            if enable_training:
                if False:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                 on_trace_ready=torch.profiler.tensorboard_trace_handler(
                                     str(runs_path / ('trace_' + str(hyperparameter_options)))),
                                 schedule=schedule(wait=2, warmup=3, active=30, repeat=1),
                                 profile_memory=True,
                                 with_stack=True) as prof:
                                 #schedule=schedule(wait=5, warmup=5, active=20),
                                 #on_trace_ready=trace_handler) as prof:
                        train_loss, train_accuracy, train_delta_diff_xy, train_delta_diff_xyz = \
                            train_or_test_SL_epoch(train_dataloader, model, optimizer, scaler, True, prof)
                    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
                    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
                    #prof.export_chrome_trace("/raid/durst/csknow/learn_bot/traces/trace_all.json")
                    quit(0)
                else:
                    train_loss, train_accuracy, train_delta_diff_xy, train_delta_diff_xyz = \
                        train_or_test_SL_epoch(train_dataloader, model, optimizer, scaler, True)
            else:
                with torch.no_grad():
                    train_loss, train_accuracy, train_delta_diff_xy, train_delta_diff_xyz = \
                        train_or_test_SL_epoch(train_dataloader, model, optimizer, scaler, False)
            with torch.no_grad():
                test_loss, test_accuracy, test_delta_diff_xy, test_delta_diff_xyz = \
                    train_or_test_SL_epoch(test_dataloader, model, None, None, False)
            cur_test_loss_float = test_loss.total_accumulator
            if cur_test_loss_float < min_test_loss:
                save_model(False, total_epochs)
                min_test_loss = cur_test_loss_float
            if True or (total_epochs + 1) % 10 == 0:
                save_model(True, total_epochs)
            save_tensorboard(train_loss, test_loss, train_accuracy, test_accuracy,
                             train_delta_diff_xy, test_delta_diff_xy, train_delta_diff_xyz, test_delta_diff_xyz,
                             total_epochs)
            total_epochs += 1

    multi_hdf5_wrapper.create_np_arrays(column_transformers)
    train_data = MultipleLatentHDF5Dataset(multi_hdf5_wrapper.train_hdf5_wrappers, column_transformers,
                                           multi_hdf5_wrapper.duplicate_last_hdf5_equal_to_rest)
    #print(f"size all train data {len(train_data[[i for i in range(len(train_data))]])}")
    #exit(0)
    test_data = MultipleLatentHDF5Dataset(multi_hdf5_wrapper.test_hdf5_wrappers, column_transformers,
                                          multi_hdf5_wrapper.duplicate_last_hdf5_equal_to_rest)
    batch_size = min(hyperparameter_options.batch_size, min(len(train_data), len(test_data)))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=5, shuffle=True, pin_memory=True)

    print(f"num train examples: {len(train_data)}")
    print(f"num test examples: {len(test_data)}")

    for X, Y, _, _, _ in train_dataloader:
        print(f"Train shape of X: {X.shape} {X.dtype}")
        print(f"Train shape of Y: {Y.shape} {Y.dtype}")
        break

    train_and_test_SL(model, train_dataloader, test_dataloader, hyperparameter_options.num_epochs())
    return train_checkpoint_paths


latent_id_cols = ['id', round_id_column, test_success_col]

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
    #small_good_rounds=[small_human_good_rounds],
    #similarity_dfs=[load_hdf5_to_pd(all_human_vs_small_human_similarity_hdf5_data_path)],
    #hand_labeled_push_round_ids=[all_human_28_second_filter_good_rounds],
    #similarity_dfs=[load_hdf5_to_pd(all_human_vs_human_28_similarity_hdf5_data_path)],
    load_similarity_data=True,
    train_test_split_file_name=train_test_split_file_name
)


def run_single_training():
    global total_epochs
    load_data_result = LoadDataResult(load_data_options)
    if len(sys.argv) > 1:
        hyperparameter_indices = [int(i) for i in sys.argv[1].split(",")]
        for index in hyperparameter_indices:
            total_epochs = 0
            hyperparameter_options = hyperparameter_option_range[index]
            hyperparameter_options.comment = load_data_result.dataset_comment
            train(TrainType.DeltaPos, load_data_result.multi_hdf5_wrapper, hyperparameter_options,
                  diff_train_test=load_data_result.diff_train_test)
    else:
        hyperparameter_options = HyperparameterOptions(comment=load_data_result.dataset_comment)
        train(TrainType.DeltaPos, load_data_result.multi_hdf5_wrapper, hyperparameter_options,
              diff_train_test=load_data_result.diff_train_test)
                             #flip_columns=[ColumnsToFlip(" CT 0", " CT 1")])


use_curriculum_training = False


def run_curriculum_training():
    global total_epochs
    # leaving this in for now, but should replace this with better mixing
    assert False
    bot_data = HDF5Wrapper(manual_latent_team_hdf5_data_path, ['id', round_id_column, game_id_column, test_success_col])
    bot_data.limit(team_data.id_df[test_success_col] == 1.)
    human_data = HDF5Wrapper(human_latent_team_hdf5_data_path, ['id', round_id_column, test_success_col])
    with open(default_selected_retake_rounds_path, "r") as f:
        good_retake_rounds = eval(f.read())
    human_data.limit(human_data.id_df[round_id_column].isin(good_retake_rounds))
    hyperparameter_options = default_hyperparameter_options
    if len(sys.argv) > 1:
        # will fix this later, for now not going to support hyperparameter search with curriculum training
        assert False
        hyperparameter_indices = [int(i) for i in sys.argv[1].split(",")]
        for index in hyperparameter_indices:
            total_epochs = 0
            hyperparameter_options = hyperparameter_option_range[index]
            train(TrainType.DeltaPos, bot_data, hyperparameter_options)
            train(TrainType.DeltaPos, human_data, hyperparameter_options,
                  load_model_path=checkpoints_path / "delta_pos_checkpoint.pt")
    else:
        just_bot_hyperparameter_options = HyperparameterOptions(comment=just_bot_comment + curriculum_comment)
        just_bot_checkpoint_paths = train(TrainType.DeltaPos, bot_data, just_bot_hyperparameter_options)
        #bot_and_human_hyperparameter_options = HyperparameterOptions(comment=bot_and_human_comment)
        #bot_and_human_checkpoint_paths = train(TrainType.DeltaPos, bot_data, bot_and_human_hyperparameter_options,
        #                                       load_model_path=just_bot_checkpoint_paths.last_not_best_path / "delta_pos_checkpoint.pt",
        #                                       secondary_data_hdf5=human_data)
        just_human_hyperparameter_options = HyperparameterOptions(comment=just_human_comment + curriculum_comment + "_no_train")
        train(TrainType.DeltaPos, human_data, just_human_hyperparameter_options,
              load_model_path=just_bot_checkpoint_paths.last_not_best_path / "delta_pos_checkpoint.pt")
              #load_model_path=checkpoints_path / "06_21_2023__11_33_23_e_60_b_512_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_t_5_c_just_bot/not_best/59" / "delta_pos_checkpoint.pt",
              #enable_training=False)
              #load_model_path=bot_and_human_checkpoint_paths.last_not_best_path / "delta_pos_checkpoint.pt")


if __name__ == "__main__":
    if use_curriculum_training:
        run_curriculum_training()
    else:
        run_single_training()


