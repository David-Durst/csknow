# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import os
from enum import Enum
from typing import Dict
import sys

import torch.optim
from torch import nn, autocast
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from learn_bot.latent.analyze.comparison_column_names import human_good_rounds, \
    all_human_vs_small_human_similarity_hdf5_data_path
from learn_bot.latent.dataset import *
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.latent_hdf5_dataset import MultipleLatentHDF5Dataset
from learn_bot.latent.place_area.load_data import human_latent_team_hdf5_data_path, manual_latent_team_hdf5_data_path, \
    LoadDataResult, LoadDataOptions
from learn_bot.latent.place_area.pos_abs_delta_conversion import delta_pos_grid_num_cells
from learn_bot.latent.place_area.column_names import place_area_input_column_types, delta_pos_output_column_types, test_success_col
from learn_bot.latent.profiling import profile_latent_model
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper
from learn_bot.libs.io_transforms import CUDA_DEVICE_STR
from learn_bot.latent.accuracy_and_loss import compute_loss, compute_accuracy_and_delta_diff, \
    finish_accuracy_and_delta_diff, \
    CPU_DEVICE_STR, LatentLosses, duplicated_name_str
from learn_bot.libs.multi_hdf5_wrapper import MultiHDF5Wrapper
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

checkpoints_path = Path(__file__).parent / 'checkpoints'
plot_path = Path(__file__).parent / 'distributions'
good_retake_rounds_path = Path(__file__).parent / 'vis' / 'good_retake_round_ids.txt'

now = datetime.now()
now_str = now.strftime("%m_%d_%Y__%H_%M_%S")
runs_path = Path(__file__).parent / 'runs'

time_model = False

@dataclass(frozen=True)
class TrainResult:
    train_dataset: LatentDataset
    test_dataset: LatentDataset
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    column_transformers: IOColumnTransformers
    model: nn.Module


class TrainType(Enum):
    DeltaPos = 1


@dataclass
class HyperparameterOptions:
    num_epochs: int = 60
    batch_size: int = 512
    learning_rate: float = 4e-5
    weight_decay: float = 0.
    layers: int = 2
    heads: int = 4
    noise_var: float = 20.
    comment: str = ""

    def to_str(self, model: TransformerNestedHiddenLatentModel):
        return f"{now_str}_e_{self.num_epochs}_b_{self.batch_size}_lr_{self.learning_rate}_wd_{self.weight_decay}_" \
               f"l_{self.layers}_h_{self.heads}_n_{self.noise_var}_t_{model.num_time_steps}_" \
               f"c_{self.comment}"

    def get_checkpoints_path(self, model: TransformerNestedHiddenLatentModel) -> Path:
        return checkpoints_path / self.to_str(model)


default_hyperparameter_options = HyperparameterOptions()
hyperparameter_option_range = [HyperparameterOptions(weight_decay=0.1),
                               HyperparameterOptions(weight_decay=1e-3),
                               HyperparameterOptions(layers=4, heads=8),
                               HyperparameterOptions(batch_size=256),
                               HyperparameterOptions(num_epochs=1024),
                               HyperparameterOptions(num_epochs=2048),
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
        column_transformers = IOColumnTransformers(place_area_input_column_types, delta_pos_output_column_types,
                                                   multi_hdf5_wrapper.train_hdf5_wrappers[0].sample_df)
        model = TransformerNestedHiddenLatentModel(column_transformers, 2 * max_enemies, delta_pos_grid_num_cells,
                                                   hyperparameter_options.layers, hyperparameter_options.heads)
        if load_model_path:
            model_file = torch.load(load_model_path)
            model.load_state_dict(model_file['model_state_dict'])
        model = model.to(device)
        input_column_types = place_area_input_column_types
        output_column_types = delta_pos_output_column_types
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

    run_checkpoints_path = hyperparameter_options.get_checkpoints_path(model)
    run_checkpoints_path.mkdir(parents=True, exist_ok=True)
    train_checkpoint_paths.best_path = run_checkpoints_path

    # define losses
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameter_options.learning_rate,
                                 weight_decay=hyperparameter_options.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # train and test the model
    first_row: torch.Tensor = None

    def train_or_test_SL_epoch(dataloader, model, optimizer, scaler, train=True, profiler=None):
        nonlocal first_row
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
        prior_bad_X = None
        prior_bad_Y = None
        prior_bad_duplicated_last = None
        prior_bad_indices = None
        with tqdm(total=len(dataloader), disable=False) as pbar:
            for batch, (X, Y, duplicated_last, indices) in enumerate(dataloader):
                batch_num += 1
                if first_row is None:
                    first_row = X[0:1, :]
                if prior_bad_X is None:
                    X, Y, duplicated_last = X.to(device), Y.to(device), duplicated_last.to(device)
                    Y = Y.float()
                else:
                    X = prior_bad_X
                    Y = prior_bad_Y
                    duplicated_last = prior_bad_duplicated_last
                    indices = prior_bad_indices
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
                with autocast(device, enabled=True):
                    pred = model(X, Y)
                    model.noise_var = -1.
                    if torch.isnan(X).any():
                        print('bad X')
                        sys.exit(0)
                    if torch.isnan(Y).any():
                        print('bad Y')
                        sys.exit(0)
                    if torch.isnan(pred[0]).any():
                        print(X)
                        print(pred[0])
                        print(torch.isnan(pred[0]).nonzero())
                        bad_batch_row = torch.isnan(pred[0]).nonzero()[0,0].item()
                        print(indices[bad_batch_row])
                        print('bad pred')
                        prior_bad_X = X.detach()
                        prior_bad_Y = Y.detach()
                        prior_bad_duplicated_last = duplicated_last.detach()
                        prior_bad_indices = indices
                        sys.exit(0)
                    batch_loss = compute_loss(pred, Y, duplicated_last, model.num_players)
                    # uncomment here and below causes memory issues
                    cumulative_loss += batch_loss
                    #losses.append(batch_loss.total_loss.tolist()[0])

                # Backpropagation
                if train:
                    scaler.scale(batch_loss.total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                compute_accuracy_and_delta_diff(pred, Y, duplicated_last, accuracy, delta_diff_xy, delta_diff_xyz,
                                                valids_per_accuracy_column, model.num_players, column_transformers)
                pbar.update(1)
                if profiler is not None:
                    profiler.step()

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
            'diff_test_train': diff_train_test
        }, model_path)
        with torch.no_grad():
            model.eval()
            script_model = torch.jit.trace(model.to(CPU_DEVICE_STR), first_row)
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

    cur_runs_path = runs_path / hyperparameter_options.to_str(model)
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
                def trace_handler(p):
                    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
                    print(output)
                    p.export_chrome_trace("/raid/durst/csknow/learn_bot/traces/trace_" + str(p.step_num) + ".json")
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=trace_handler) as prof:
                    train_loss, train_accuracy, train_delta_diff_xy, train_delta_diff_xyz = \
                        train_or_test_SL_epoch(train_dataloader, model, optimizer, scaler, True, prof)
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
            if (total_epochs + 1) % 10 == 0:
                save_model(True, total_epochs)
            save_tensorboard(train_loss, test_loss, train_accuracy, test_accuracy,
                             train_delta_diff_xy, test_delta_diff_xy, train_delta_diff_xyz, test_delta_diff_xyz,
                             total_epochs)
            total_epochs += 1

    multi_hdf5_wrapper.create_np_arrays(column_transformers)
    train_data = MultipleLatentHDF5Dataset(multi_hdf5_wrapper.train_hdf5_wrappers, column_transformers,
                                           multi_hdf5_wrapper.duplicate_last_hdf5_equal_to_rest)
    test_data = MultipleLatentHDF5Dataset(multi_hdf5_wrapper.test_hdf5_wrappers, column_transformers,
                                          multi_hdf5_wrapper.duplicate_last_hdf5_equal_to_rest)
    batch_size = min(hyperparameter_options.batch_size, min(len(train_data), len(test_data)))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)

    print(f"num train examples: {len(train_data)}")
    print(f"num test examples: {len(test_data)}")

    for X, Y, _, _ in train_dataloader:
        print(f"Train shape of X: {X.shape} {X.dtype}")
        print(f"Train shape of Y: {Y.shape} {Y.dtype}")
        break

    train_and_test_SL(model, train_dataloader, test_dataloader, hyperparameter_options.num_epochs)
    return train_checkpoint_paths


latent_id_cols = ['id', round_id_column, test_success_col]

load_data_options = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=False,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_all_human_data=True,
    add_manual_to_all_human_data=True,
    limit_manual_data_to_no_enemies_nav=True,
    small_good_rounds=human_good_rounds,
    similarity_df=load_hdf5_to_pd(all_human_vs_small_human_similarity_hdf5_data_path)
)


def run_single_training():
    load_data_result = LoadDataResult(load_data_options)
    if len(sys.argv) > 1:
        hyperparameter_indices = [int(i) for i in sys.argv[1].split(",")]
        for index in hyperparameter_indices:
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
    with open(good_retake_rounds_path, "r") as f:
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


