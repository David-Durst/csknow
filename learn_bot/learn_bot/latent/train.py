# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
from enum import Enum
from typing import Dict
import sys

import pandas as pd
import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from learn_bot.latent.aggression.column_names import aggression_input_column_types, aggression_output_column_types
from learn_bot.latent.aggression.latent_to_distributions import get_aggression_distributions, num_aggression_options, \
    get_aggression_probability
from learn_bot.latent.dataset import *
from learn_bot.latent.engagement.column_names import round_id_column, engagement_input_column_types, engagement_output_column_types
from learn_bot.latent.engagement.latent_to_distributions import get_engagement_target_distributions, num_target_options, \
    get_engagement_probability
from learn_bot.latent.latent_hdf5_dataset import LatentHDF5Dataset
from learn_bot.latent.lstm_latent_model import LSTMLatentModel
from learn_bot.latent.mlp_hidden_latent_model import MLPHiddenLatentModel
from learn_bot.latent.mlp_latent_model import MLPLatentModel
from learn_bot.latent.mlp_nested_hidden_latent_model import MLPNestedHiddenLatentModel
from learn_bot.latent.place_area.create_test_data import create_left_right_train_data, create_left_right_test_data
from learn_bot.latent.place_area.pos_abs_delta_conversion import delta_pos_grid_num_cells
from learn_bot.latent.order.latent_to_distributions import get_order_probability
from learn_bot.latent.place_area.column_names import place_area_input_column_types, place_output_column_types, \
    num_places, area_grid_size, area_output_column_types, delta_pos_output_column_types, test_success_col
from learn_bot.latent.profiling import profile_latent_model
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd, HDF5Wrapper, PDWrapper
from learn_bot.libs.io_transforms import CUDA_DEVICE_STR
from learn_bot.latent.accuracy_and_loss import compute_loss, compute_accuracy, finish_accuracy, \
    CPU_DEVICE_STR, LatentLosses
from learn_bot.libs.plot_features import plot_untransformed_and_transformed
from learn_bot.libs.df_grouping import train_test_split_by_col, make_index_column, TrainTestSplit, get_test_col_ids
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Optional

checkpoints_path = Path(__file__).parent / 'checkpoints'
plot_path = Path(__file__).parent / 'distributions'

now = datetime.now()
runs_path = Path(__file__).parent / 'runs' / now.strftime("%m_%d_%Y__%H_%M_%S")


def path_append(p: Path, suffix: str) -> Path:
    return p.parent / (p.name + suffix)


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
    num_epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 4e-5
    weight_decay: float = 0.
    layers: int = 2
    heads: int = 4
    noise_var: float = 20.

    def __str__(self):
        return f"e_{self.num_epochs}_b_{self.batch_size}_lr_{self.learning_rate}_wd_{self.weight_decay}_l_{self.layers}_h_{self.heads}_n_{self.noise_var}"


default_hyperparameter_options = HyperparameterOptions()
hyperparameter_option_range = [HyperparameterOptions(layers=4, heads=8),
                               HyperparameterOptions(num_epochs=3000, batch_size=512, learning_rate=1e-6),
                               HyperparameterOptions(num_epochs=3000, batch_size=512, learning_rate=5e-7),
                               HyperparameterOptions(num_epochs=3000, batch_size=512, learning_rate=1e-7)]
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


def train(train_type: TrainType, all_data_hdf5: HDF5Wrapper, hyperparameter_options: HyperparameterOptions = default_hyperparameter_options,
          diff_train_test=True, flip_columns: List[ColumnsToFlip] = [], force_test_hdf5: Optional[HDF5Wrapper] = None):

    run_checkpoints_path = checkpoints_path
    if hyperparameter_options != default_hyperparameter_options:
        run_checkpoints_path = run_checkpoints_path / str(hyperparameter_options)
        run_checkpoints_path.mkdir(parents=True, exist_ok=True)


    if diff_train_test:
        train_test_split = train_test_split_by_col(all_data_hdf5.id_df, round_id_column)
        train_hdf5 = all_data_hdf5.clone()
        train_hdf5.limit(train_test_split.train_predicate)
        train_group_ids = train_test_split.train_group_ids
        test_hdf5 = all_data_hdf5.clone()
        test_hdf5.limit(~train_test_split.train_predicate)
        test_group_ids = get_test_col_ids(train_test_split, round_id_column)
    else:
        train_hdf5 = all_data_hdf5
        train_group_ids = list(all_data_hdf5.id_df.loc[:, round_id_column].unique())
        if force_test_hdf5:
            test_hdf5 = force_test_hdf5
            test_group_ids = list(force_test_hdf5.id_df.loc[:, round_id_column].unique())
        else:
            test_hdf5 = all_data_hdf5
            test_group_ids = train_group_ids

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
                                                   train_hdf5.sample_df)
        model = TransformerNestedHiddenLatentModel(column_transformers, 2 * max_enemies, delta_pos_grid_num_cells,
                                                   hyperparameter_options.layers, hyperparameter_options.heads).to(device)
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

    # define losses
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameter_options.learning_rate,
                                 weight_decay=hyperparameter_options.weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # train and test the model
    first_row: torch.Tensor = None

    def train_or_test_SL_epoch(dataloader, model, optimizer, train=True):
        nonlocal first_row
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        if train:
            model.train()
        else:
            model.eval()
        cumulative_loss = LatentLosses()
        accuracy = {}
        valids_per_accuracy_column = {}
        #losses = []
        # bar = Bar('Processing', max=size)
        with tqdm(total=len(dataloader), disable=False) as pbar:
            for batch, (X, Y) in enumerate(dataloader):
                if first_row is None:
                    first_row = X[0:1, :]
                X, Y = X.to(device), Y.to(device)
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
                    print('bad pred')
                    sys.exit(0)
                batch_loss = compute_loss(pred, Y, column_transformers)
                cumulative_loss += batch_loss
                #losses.append(batch_loss.get_total_loss().tolist()[0])

                # Backpropagation
                if train:
                    optimizer.zero_grad()
                    batch_loss.get_total_loss().backward()
                    optimizer.step()

                compute_accuracy(pred, Y, accuracy, valids_per_accuracy_column, column_transformers)
                pbar.update(1)

        cumulative_loss /= len(dataloader)
        for name in column_transformers.output_types.column_names():
            if name in valids_per_accuracy_column and valids_per_accuracy_column[name] > 0:
                accuracy[name] = accuracy[name].item() / valids_per_accuracy_column[name].item()
        accuracy_string = finish_accuracy(accuracy, valids_per_accuracy_column, column_transformers)
        train_test_str = "Train" if train else "Test"
        print(f"Epoch {train_test_str} Accuracy: {accuracy_string}, Transformed Avg Loss: {cumulative_loss.get_total_loss().item():>8f}")
        return cumulative_loss, accuracy

    def save_model():
        nonlocal train_type
        if train_type == TrainType.DeltaPos:
            model_path = run_checkpoints_path / 'delta_pos_checkpoint.pt'
        torch.save({
            'train_group_ids': train_group_ids,
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
            test_group_ids_str = ",".join([str(round_id) for round_id in test_group_ids])
            if train_type == TrainType.DeltaPos:
                script_model.save(run_checkpoints_path / 'delta_pos_script_model.pt')
                with open(run_checkpoints_path / 'delta_pos_test_round_ids.csv', 'w+') as f:
                    f.write(test_group_ids_str)
            model.to(device)

    cur_runs_path = path_append(runs_path, "_" + str(hyperparameter_options))
    writer = SummaryWriter(cur_runs_path)
    def save_tensorboard(train_loss: LatentLosses, test_loss: LatentLosses, train_accuracy: Dict, test_accuracy: Dict,
                         epoch_num):
        train_loss.add_scalars(writer, 'train', epoch_num)
        test_loss.add_scalars(writer, 'test', epoch_num)
        for name, acc in train_accuracy.items():
            writer.add_scalar('train/acc/' + name, acc, epoch_num)
        for name, acc in test_accuracy.items():
            writer.add_scalar('test/acc/' + name, acc, epoch_num)

    min_test_loss = float("inf")
    def train_and_test_SL(model, train_dataloader, test_dataloader, num_epochs):
        nonlocal optimizer, min_test_loss
        for epoch_num in range(num_epochs):
            print(f"\nEpoch {epoch_num}\n" + f"-------------------------------")
            train_loss, train_accuracy = train_or_test_SL_epoch(train_dataloader, model, optimizer, True)
            with torch.no_grad():
                test_loss, test_accuracy = train_or_test_SL_epoch(test_dataloader, model, None, False)
            cur_test_less_float = test_loss.get_total_loss().item()
            if cur_test_less_float < min_test_loss:
                save_model()
                min_test_loss = cur_test_less_float
            save_tensorboard(train_loss, test_loss, train_accuracy, test_accuracy, epoch_num)

    train_hdf5.create_np_array(column_transformers)
    if not diff_train_test and force_test_hdf5:
        test_hdf5.create_np_array(column_transformers)
    train_data = LatentHDF5Dataset(train_hdf5, column_transformers)
    test_data = LatentHDF5Dataset(test_hdf5, column_transformers)
    batch_size = min(hyperparameter_options.batch_size, min(len(train_hdf5), len(test_hdf5)))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=10, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=10, shuffle=True, pin_memory=True)

    print(f"num train examples: {len(train_data)}")
    print(f"num test examples: {len(test_data)}")

    for X, Y in train_dataloader:
        print(f"Train shape of X: {X.shape} {X.dtype}")
        print(f"Train shape of Y: {Y.shape} {Y.dtype}")
        break

    train_and_test_SL(model, train_dataloader, test_dataloader, hyperparameter_options.num_epochs)


latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'
small_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'smallBehaviorTreeTeamFeatureStore.parquet'
manual_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'manual_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'
manual_rounds_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'saved_datasets' / 'bot_sample_traces_5_10_23_ticks.csv'
rollout_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'

use_manual_data = False
use_test_data = False

def run_team_analysis():
    read_start = time.time()
    diff_train_test = True
    test_team_data = None
    if use_manual_data:
        team_data = HDF5Wrapper(manual_latent_team_hdf5_data_path, ['id', round_id_column, test_success_col])
        team_data.limit(team_data.id_df[test_success_col] == 1.)
    elif use_test_data:
        base_data = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=[i for i in range(1)])
        team_data_df = create_left_right_train_data(base_data)
        team_data = PDWrapper('train', team_data_df, ['id', round_id_column, test_success_col])
        test_team_data_df = create_left_right_test_data(base_data)
        test_team_data = PDWrapper('test', test_team_data_df, ['id', round_id_column, test_success_col])
        diff_train_test = False
    else:
        team_data = HDF5Wrapper(latent_team_hdf5_data_path, ['id', round_id_column, test_success_col])
    hyperparameter_options = default_hyperparameter_options
    if len(sys.argv) > 1:
        hyperparameter_indices = [int(i) for i in sys.argv[1].split(",")]
        for index in hyperparameter_indices:
            hyperparameter_options = hyperparameter_option_range[index]
            train(TrainType.DeltaPos, team_data, hyperparameter_options, diff_train_test=diff_train_test)
    else:
        train(TrainType.DeltaPos, team_data, diff_train_test=diff_train_test, force_test_hdf5=test_team_data)
                             #flip_columns=[ColumnsToFlip(" CT 0", " CT 1")])


if __name__ == "__main__":
    run_team_analysis()

