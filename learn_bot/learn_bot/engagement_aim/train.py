# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from dataset import *
from learn_bot.libs.accuracy_and_loss import compute_loss, compute_accuracy, finish_accuracy, CUDA_DEVICE_STR, \
    CPU_DEVICE_STR
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes, PRIOR_TICKS, FUTURE_TICKS, CUR_TICK
from learn_bot.engagement_aim.lstm_aim_model import LSTMAimModel
from learn_bot.engagement_aim.output_plotting import plot_untransformed_and_transformed, ModelOutputRecording
from learn_bot.libs.df_grouping import train_test_split_by_col
from typing import List, Dict, Deque
from dataclasses import dataclass
from collections import deque
from alive_progress import alive_bar
from learn_bot.libs.profiling import *

from learn_bot.libs.temporal_column_names import TemporalIOColumnNames

def train():
    all_data_df = pd.read_csv(Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'engagementAim.csv')

    #all_data_df = all_data_df[all_data_df['num shots fired'] > 0]

    train_test_split = train_test_split_by_col(all_data_df, 'engagement id')
    train_df = train_test_split.train_df
    test_df = train_test_split.test_df

    base_float_columns: List[str] = ["delta view angle x", "delta view angle y",
                                     "recoil angle x", "recoil angle y",
                                     "delta view angle recoil adjusted x", "delta view angle recoil adjusted y",
                                     "delta position x", "delta position y", "delta position z",
                                     "eye-to-head distance"]

    temporal_io_float_column_names = TemporalIOColumnNames(base_float_columns, PRIOR_TICKS, CUR_TICK, FUTURE_TICKS)

    non_temporal_float_columns = ["num shots fired", "ticks since last fire"]

    input_categorical_columns: List[str] = ["weapon type"]

    # transform input and output
    input_column_types = ColumnTypes(temporal_io_float_column_names.input_columns + non_temporal_float_columns,
                                    input_categorical_columns, [6])

    output_column_types = ColumnTypes(temporal_io_float_column_names.output_columns, [], [])

    column_transformers = IOColumnTransformers(input_column_types, output_column_types, all_data_df)

    # plot data set with and without transformers
    plot_untransformed_and_transformed('train+test labels', all_data_df, temporal_io_float_column_names.vis_columns + non_temporal_float_columns,
                                       input_categorical_columns)

    # Get cpu or gpu device for training.
    device: str = CUDA_DEVICE_STR if torch.cuda.is_available() else CPU_DEVICE_STR
    print(f"Using {device} device")

    # Define model
    embedding_dim = 5
    #model = MLPAimModel(column_transformers).to(device)
    model = LSTMAimModel(column_transformers,
                         len(temporal_io_float_column_names.input_columns), len(non_temporal_float_columns)).to(device)
    print(model)
    params = list(model.parameters())
    print("params by layer")
    for param_layer in params:
        print(param_layer.shape)

    # define losses
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    output_cols = column_transformers.output_types.column_names()

    # train and test the model
    first_row: torch.Tensor
    model_output_recording: ModelOutputRecording = ModelOutputRecording(model)
    def train_or_test_SL_epoch(dataloader, model, optimizer, epoch_num, train = True):
        nonlocal first_row, model_output_recording
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        if train:
            model.train()
        else:
            model.eval()
        cumulative_loss = 0
        accuracy = {}
        #bar = Bar('Processing', max=size)
        for name in column_transformers.output_types.column_names():
            accuracy[name] = 0
        for batch, (X, Y) in enumerate(dataloader):
            if batch == 0 and epoch_num == 0 and train:
                first_row = X[0:1,:]
            X, Y = X.to(device), Y.to(device)
            transformed_Y = column_transformers.transform_columns(False, Y)
            #XR = torch.randn_like(X, device=device)
            #XR[:,0] = X[:,0]
            #YZ = torch.zeros_like(Y) + 0.1

            # Compute prediction error
            pred = model(X)
            batch_loss = compute_loss(pred, transformed_Y, column_transformers)
            cumulative_loss += batch_loss

            # Backpropagation
            if train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            if False and train and batch % 100 == 0:
                loss, current = batch_loss.item(), batch * len(X)
                print('pred')
                print(pred[0:2])
                print('y')
                print(Y[0:2])
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            compute_accuracy(pred, Y, accuracy, column_transformers)
            if epoch_num == epochs - 1:
                model_output_recording.record_output(pred, Y, transformed_Y, train)
            #bar.next(X.shape[0])

        cumulative_loss /= num_batches
        for name in column_transformers.output_types.column_names():
            accuracy[name] /= size
        accuracy_string = finish_accuracy(accuracy, column_transformers)
        train_test_str = "Train" if train else "Test"
        print(f"Epoch {train_test_str} Accuracy: {accuracy_string}, Transformed Avg Loss: {cumulative_loss:>8f}")


    epochs = 1
    def train_and_test_SL(model, train_dataloader, test_dataloader):
        for epoch_num in range(epochs):
            print(f"\nEpoch {epoch_num+1}\n-------------------------------")
            train_or_test_SL_epoch(train_dataloader, model, optimizer, epoch_num, True)
            train_or_test_SL_epoch(test_dataloader, model, None, epoch_num, False)

    @dataclass
    class PolicyOutput:
        delta_view_angle_x: float
        delta_view_angle_y: float

    class PolicyHistory:
        # generate the input tensor for the next policy iteration
        # create the dict for inserting a new training data point into the data frame
        def get_x_field_str(self, tick: int = -1):
            return f"delta view angle x (t-{abs(tick)})"
        def get_y_field_str(self, tick: int = -1):
            return f"delta view angle y (t-{abs(tick)})"

        row_dict: Dict
        input_tensor: torch.Tensor

        def __init__(self, row_dict: Dict, input_tensor: torch.Tensor):
            self.row_dict = row_dict
            self.input_tensor = input_tensor

        def add_row(self, policy_output: PolicyOutput, model: LSTMAimModel, new_row_dict: Dict,
                    new_input_tensor: torch.Tensor, agg_dicts: List[Dict]):
            # update new input_tensor and row_dict by setting the view angles from old input_tensor
            # most recent values are form policy_output
            for i in range(PRIOR_TICKS, -1):
                new_row_dict[self.get_x_field_str(i)] = self.row_dict[self.get_x_field_str(i+1)]
                new_row_dict[self.get_y_field_str(i)] = self.row_dict[self.get_y_field_str(i+1)]

                model.set_untransformed_output(new_input_tensor, self.get_x_field_str(i),
                                               model.get_untransformed_output(self.input_tensor, self.get_x_field_str(i+1)))
                model.set_untransformed_output(new_input_tensor, self.get_y_field_str(i),
                                               model.get_untransformed_output(self.input_tensor, self.get_y_field_str(i+1)))

            new_row_dict[self.get_x_field_str()] = policy_output.delta_view_angle_x
            new_row_dict[self.get_y_field_str()] = policy_output.delta_view_angle_y
            model.set_untransformed_output(new_input_tensor, self.get_x_field_str(), policy_output.delta_view_angle_x)
            model.set_untransformed_output(new_input_tensor, self.get_y_field_str(), policy_output.delta_view_angle_y)

            self.row_dict = new_row_dict
            self.input_tensor = new_input_tensor
            agg_dicts.append(self.row_dict)

    def on_policy_inference(dataset: AimDataset, orig_df: pd.DataFrame, model: LSTMAimModel) -> pd.DataFrame:
        agg_dicts = []
        inner_agg_df = None
        model.eval()
        prior_row_round_id = -1
        # this tracks history so it can produce inputs
        history_per_engagement: Dict[int, PolicyHistory] = {}
        # this tracks last output, as output only used as input when hit next input
        last_output_per_engagement: Dict[int, PolicyOutput] = {}
        with torch.no_grad():
            with alive_bar(len(dataset), force_tty=True) as bar:
                for i in range(len(dataset)):
                    if prior_row_round_id != dataset.round_id.iloc[i]:
                        round_df = pd.DataFrame.from_dict(agg_dicts)
                        if inner_agg_df is not None:
                            inner_agg_df = pd.concat([inner_agg_df, round_df], ignore_index=True)
                        else:
                            inner_agg_df = round_df
                        agg_dicts = []
                        history_per_engagement = {}
                        last_output_per_engagement = {}
                    prior_row_round_id = dataset.round_id.iloc[i]
                    engagement_id = dataset.engagement_id.iloc[i]
                    if engagement_id in history_per_engagement:
                        history_per_engagement[engagement_id].add_row(
                            last_output_per_engagement[engagement_id],
                            model,
                            orig_df.iloc[i].to_dict(),
                            torch.unsqueeze(dataset[i][0], dim=0).detach(),
                            agg_dicts
                        )
                    else:
                        history_per_engagement[engagement_id] = PolicyHistory(
                            orig_df.iloc[i].to_dict(), torch.unsqueeze(dataset[i][0], dim=0).detach())
                    X_rolling = history_per_engagement[engagement_id].input_tensor
                    pred = model(X_rolling.to(CUDA_DEVICE_STR)).to(CPU_DEVICE_STR).detach()
                    # need to add output to data set
                    last_output_per_engagement[engagement_id] = PolicyOutput(
                        model.get_untransformed_output(pred, "delta view angle x (t)"),
                        model.get_untransformed_output(pred, "delta view angle y (t)")
                    )
                    bar()
        return inner_agg_df

    agg_df = None
    total_train_df = train_df
    dad_iters = 4
    for dad_num in range(dad_iters):
        print(f"DaD Iter {dad_num + 1}\n-------------------------------")
        # step 1: train model
        # create data sets for pytorch
        training_data = AimDataset(total_train_df, column_transformers)
        test_data = AimDataset(test_df, column_transformers)

        batch_size = 64

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        print(f"num train examples: {len(training_data)}")
        print(f"num test examples: {len(test_data)}")

        if dad_num == 0:
            for X, Y in train_dataloader:
                print(f"Train shape of X: {X.shape} {X.dtype}")
                print(f"Train shape of Y: {Y.shape} {Y.dtype}")
                break

            for X, Y in test_dataloader:
                print(f"Test shape of X: {X.shape} {X.dtype}")
                print(f"Test shape of Y: {Y.shape} {Y.dtype}")
                break

        train_and_test_SL(model, train_dataloader, test_dataloader)

        # step 2: inference and result collection
        new_agg_df = on_policy_inference(training_data, train_df, model)

        # step 3: create new training data set
        if agg_df is None:
            agg_df = new_agg_df
        else:
            agg_df = pd.concat([agg_df, new_agg_df], ignore_index=True)
        total_train_df = pd.concat([train_df, new_agg_df], ignore_index=True)




    model_output_recording.plot(column_transformers, temporal_io_float_column_names.vis_columns)

    script_model = torch.jit.trace(model.to(CPU_DEVICE_STR), first_row)
    script_model.save(Path(__file__).parent / '..' / '..' / 'models' / 'engagement_aim_model' / 'script_model.pt')

    print("Done")

if __name__ == "__main__":
    train()