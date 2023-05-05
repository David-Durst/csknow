# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
from enum import Enum
from typing import Dict
import sys

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
from learn_bot.latent.lstm_latent_model import LSTMLatentModel
from learn_bot.latent.mlp_hidden_latent_model import MLPHiddenLatentModel
from learn_bot.latent.mlp_latent_model import MLPLatentModel
from learn_bot.latent.mlp_nested_hidden_latent_model import MLPNestedHiddenLatentModel
from learn_bot.latent.order.column_names import order_input_column_types, order_output_column_types, num_orders_per_site, \
    PlayerOrderColumns
from learn_bot.latent.order.latent_to_distributions import get_order_probability
from learn_bot.latent.place_area.column_names import place_area_input_column_types, place_output_column_types, \
    num_places, area_grid_size, area_output_column_types
from learn_bot.latent.place_area.latent_to_distributions import get_place_area_probability
from learn_bot.latent.profiling import profile_latent_model
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
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
    Engagement = 1
    Aggression = 2
    Order = 3
    Place = 4
    Area = 5


def train(train_type: TrainType, all_data_df: pd.DataFrame, num_epochs: int,
          windowed=False, save=True) -> TrainResult:

    train_test_split = train_test_split_by_col(all_data_df, round_id_column)
    train_df = train_test_split.train_df.copy()
    train_group_ids = train_test_split.train_group_ids
    make_index_column(train_df)
    test_df = train_test_split.test_df.copy()
    test_group_ids = get_test_col_ids(train_test_split, round_id_column)
    make_index_column(test_df)

    # Get cpu or gpu device for training.
    device: str = CUDA_DEVICE_STR if torch.cuda.is_available() else CPU_DEVICE_STR
    # device = CPU_DEVICE_STR
    print(f"Using {device} device")

    # Define model
    if train_type == TrainType.Engagement:
        # transform input and output
        column_transformers = IOColumnTransformers(engagement_input_column_types, engagement_output_column_types,
                                                   train_df)
        model = MLPHiddenLatentModel(column_transformers, num_target_options, get_engagement_target_distributions).to(device)
        input_column_types = engagement_input_column_types
        output_column_types = engagement_output_column_types
        prob_func = get_engagement_probability
    elif train_type == TrainType.Aggression:
        column_transformers = IOColumnTransformers(aggression_input_column_types, aggression_output_column_types,
                                                   train_df)
        model = MLPHiddenLatentModel(column_transformers, num_aggression_options, get_aggression_distributions).to(
            device)
        input_column_types = aggression_input_column_types
        output_column_types = aggression_output_column_types
        prob_func = get_aggression_probability
    elif train_type == TrainType.Order:
        column_transformers = IOColumnTransformers(order_input_column_types, order_output_column_types,
                                                   train_df)
        model = TransformerNestedHiddenLatentModel(column_transformers, 2*max_enemies, 2*num_orders_per_site).to(device)
        input_column_types = order_input_column_types
        output_column_types = order_output_column_types
        prob_func = get_order_probability
    elif train_type == TrainType.Place:
        column_transformers = IOColumnTransformers(place_area_input_column_types, place_output_column_types,
                                                   train_df)
        model = TransformerNestedHiddenLatentModel(column_transformers, 2*max_enemies, num_places).to(device)
        input_column_types = place_area_input_column_types
        output_column_types = place_output_column_types
        prob_func = get_place_area_probability
    elif train_type == TrainType.Area:
        column_transformers = IOColumnTransformers(place_area_input_column_types, area_output_column_types,
                                                   train_df)
        model = TransformerNestedHiddenLatentModel(column_transformers, 2 * max_enemies, area_grid_size).to(device)
        input_column_types = place_area_input_column_types
        output_column_types = area_output_column_types
        prob_func = get_place_area_probability
    else:
        raise Exception("invalid train type")

    # plot data set with and without transformers
    #plot_untransformed_and_transformed(plot_path, 'train and test labels', all_data_df,
    #                                   input_column_types.float_standard_cols + output_column_types.categorical_distribution_cols_flattened,
    #                                   input_column_types.categorical_cols + output_column_types.categorical_cols)
    #model = MLPLatentModel(column_transformers).to(device)
    #model = LSTMLatentModel(column_transformers).to(device)

    print(model)
    params = list(model.parameters())
    print("params by layer")
    for param_layer in params:
        print(param_layer.shape)

    # define losses
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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
        losses = []
        # bar = Bar('Processing', max=size)
        for name in column_transformers.output_types.column_names():
            accuracy[name] = 0
        with tqdm(total=len(dataloader), disable=False) as pbar:
            for batch, (X, Y) in enumerate(dataloader):
                if first_row is None:
                    first_row = X[0:1, :]
                X, Y = X.to(device), Y.to(device)
                if windowed:
                    transformed_Y = column_transformers.nested_transform_columns(False, Y, X, window_size=window_size)
                else:
                    transformed_Y = column_transformers.transform_columns(False, Y, X)
                # XR = torch.randn_like(X, device=device)
                # XR[:,0] = X[:,0]
                # YZ = torch.zeros_like(Y) + 0.1

                if time_model:
                    if train_type == TrainType.Engagement:
                        model_path = checkpoints_path / 'engagement_script_model.pt'
                    elif train_type == TrainType.Aggression:
                        model_path = checkpoints_path / 'aggression_script_model.pt'
                    elif train_type == TrainType.Order:
                        model_path = checkpoints_path / 'order_script_model.pt'
                    elif train_type == TrainType.Place:
                        model_path = checkpoints_path / 'place_script_model.pt'
                    elif train_type == TrainType.Area:
                        model_path = checkpoints_path / 'area_script_model.pt'
                    profile_latent_model(model_path, batch_size, X)

                # Compute prediction error
                pred = model(X)
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
                batch_loss = compute_loss(X, pred, transformed_Y, Y, column_transformers, prob_func)
                cumulative_loss += batch_loss
                losses.append(batch_loss.get_total_loss().tolist()[0])

                # Backpropagation
                if train:
                    optimizer.zero_grad()
                    batch_loss.get_total_loss().backward()
                    optimizer.step()

                compute_accuracy(pred, Y, accuracy, column_transformers)
                pbar.update(1)

        cumulative_loss /= num_batches
        for name in column_transformers.output_types.column_names():
            accuracy[name] /= size
        accuracy_string = finish_accuracy(accuracy, column_transformers)
        train_test_str = "Train" if train else "Test"
        print(f"Epoch {train_test_str} Accuracy: {accuracy_string}, Transformed Avg Loss: {cumulative_loss.get_total_loss().item():>8f}")
        return cumulative_loss, accuracy

    def save_model():
        torch.save({
            'train_group_ids': train_group_ids,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'column_transformers': column_transformers,
        }, checkpoints_path / 'checkpoint.pt')

    writer = SummaryWriter(runs_path)
    def save_tensorboard(train_loss: LatentLosses, test_loss: LatentLosses, train_accuracy: Dict, test_accuracy: Dict,
                         epoch_num):
        train_loss.add_scalars(writer, 'train', epoch_num)
        test_loss.add_scalars(writer, 'test', epoch_num)
        for name, acc in train_accuracy.items():
            writer.add_scalar('train/acc/' + name, acc, epoch_num)
        for name, acc in test_accuracy.items():
            writer.add_scalar('test/acc/' + name, acc, epoch_num)

    def train_and_test_SL(model, train_dataloader, test_dataloader, num_epochs):
        nonlocal optimizer
        for epoch_num in range(num_epochs):
            print(f"\nEpoch {epoch_num}\n" + f"-------------------------------")
            train_loss, train_accuracy = train_or_test_SL_epoch(train_dataloader, model, optimizer, True)
            with torch.no_grad():
                test_loss, test_accuracy = train_or_test_SL_epoch(test_dataloader, model, None, False)
            save_model()
            save_tensorboard(train_loss, test_loss, train_accuracy, test_accuracy, epoch_num)

    train_data = LatentDataset(train_df, column_transformers, windowed=windowed)
    test_data = LatentDataset(test_df, column_transformers, windowed=windowed)
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    print(f"num train examples: {len(train_data)}")
    print(f"num test examples: {len(test_data)}")

    for X, Y in train_dataloader:
        print(f"Train shape of X: {X.shape} {X.dtype}")
        print(f"Train shape of Y: {Y.shape} {Y.dtype}")
        break

    train_and_test_SL(model, train_dataloader, test_dataloader, num_epochs)

    if save:
        with torch.no_grad():
            model.eval()
            script_model = torch.jit.trace(model.to(CPU_DEVICE_STR), first_row)
            #tmp_model = SimplifiedTransformerNestedHiddenLatentModel().to(device)
            #tmp_model.to(CPU_DEVICE_STR)
            #tmp_model.eval()
            #torch.jit.trace(tmp_model, torch.ones([64, 10, 512]))
            #  torch.jit.script(tmp_model)
            test_group_ids_str = ",".join([str(round_id) for round_id in test_group_ids])
            if train_type == TrainType.Engagement:
                script_model.save(checkpoints_path / 'engagement_script_model.pt')
                with open(checkpoints_path / 'engagement_test_round_ids.csv', 'w+') as f:
                    f.write(test_group_ids_str)
            elif train_type == TrainType.Aggression:
                script_model.save(checkpoints_path / 'aggression_script_model.pt')
                with open(checkpoints_path / 'aggression_test_round_ids.csv', 'w+') as f:
                    f.write(test_group_ids_str)
            elif train_type == TrainType.Order:
                script_model.save(checkpoints_path / 'order_script_model.pt')
                with open(checkpoints_path / 'order_test_round_ids.csv', 'w+') as f:
                    f.write(test_group_ids_str)
            elif train_type == TrainType.Place:
                script_model.save(checkpoints_path / 'place_script_model.pt')
                with open(checkpoints_path / 'place_test_round_ids.csv', 'w+') as f:
                    f.write(test_group_ids_str)
            elif train_type == TrainType.Area:
                script_model.save(checkpoints_path / 'area_script_model.pt')
                with open(checkpoints_path / 'area_test_round_ids.csv', 'w+') as f:
                    f.write(test_group_ids_str)
            model.to(device)

    return TrainResult(train_data, test_data, train_df, test_df, column_transformers, model)

latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'


def run_team_analysis():
    team_data_df = load_hdf5_to_pd(latent_team_hdf5_data_path)
    team_data_df = team_data_df[(team_data_df['valid'] == 1.) & (team_data_df['c4 status'] < 2)]
    train_result = train(TrainType.Order, team_data_df, num_epochs=3, windowed=False)
    train_result = train(TrainType.Place, team_data_df, num_epochs=3, windowed=False)
    train_result = train(TrainType.Area, team_data_df, num_epochs=3, windowed=False)


def run_individual_analysis():
    all_data_df = load_hdf5_to_pd(latent_hdf5_data_path)
    all_data_df = all_data_df[all_data_df['valid'] == 1.]
    #all_data_df = all_data_df.iloc[:500000]
    #all_data_df = load_hdf5_to_pd(latent_window_hdf5_data_path)
    train_result = train(TrainType.Engagement, all_data_df, num_epochs=1, windowed=False)
    train_result = train(TrainType.Aggression, all_data_df, num_epochs=1, windowed=False)


if __name__ == "__main__":
    run_team_analysis()
    run_individual_analysis()

# all_data_df[((all_data_df['pct nearest crosshair enemy 2s 0'] + all_data_df['pct nearest crosshair enemy 2s 1'] + all_data_df['pct nearest crosshair enemy 2s 2'] + all_data_df['pct nearest crosshair enemy 2s 3'] + all_data_df['pct nearest crosshair enemy 2s 4'] + all_data_df['pct nearest crosshair enemy 2s 5']) < 0.9) & (all_data_df['valid'] == 1)]
# all_data_df[(all_data_df['pct nearest enemy change 2s decrease'] + all_data_df['pct nearest enemy change 2s constant'] + all_data_df['pct nearest enemy change 2s increase'] < 0.9) & (all_data_df['valid'] == 1)]
# all_data_df[(all_data_df['pct nearest enemy change 2s decrease'] + all_data_df['pct nearest enemy change 2s constant'] > 0.1) & (all_data_df['valid'] == 1)]
# all_data_df[(all_data_df['fire next 2s'] + all_data_df['neg fire next 2s'] < 0.9) & (all_data_df['valid'] == 1)]
# all_data_df[(all_data_df['visible enemy 2s'] + all_data_df['neg visible enemy 2s'] < 0.9) & (all_data_df['valid'] == 1)]
# team_data_df[(team_data_df['distribution nearest a order 1 30s CT 0'] + team_data_df['distribution nearest a order 2 30s CT 0'] > 1.)]
