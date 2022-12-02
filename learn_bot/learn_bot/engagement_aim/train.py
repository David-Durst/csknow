# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch.optim
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from dataset import *
from learn_bot.engagement_aim.mlp_aim_model import MLPAimModel
from learn_bot.libs.accuracy_and_loss import compute_loss, compute_accuracy, finish_accuracy, CUDA_DEVICE_STR, \
    CPU_DEVICE_STR
from learn_bot.engagement_aim.lstm_aim_model import LSTMAimModel
from learn_bot.engagement_aim.output_plotting import plot_untransformed_and_transformed, ModelOutputRecording
from learn_bot.libs.df_grouping import train_test_split_by_col, make_index_column
from learn_bot.engagement_aim.dad import on_policy_inference, create_dad_dataset
from tqdm import tqdm
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainResult:
    train_dataset: AimDataset
    test_dataset: AimDataset
    column_transformers: IOColumnTransformers
    model: nn.Module


def train(all_data_df: pd.DataFrame, dad_iters=4, num_epochs=5, save=True,
          diff_train_test=True) -> TrainResult:
    # all_data_df = all_data_df[all_data_df['num shots fired'] > 0]

    if diff_train_test:
        train_test_split = train_test_split_by_col(all_data_df, 'engagement id')
        train_df = train_test_split.train_df
        make_index_column(train_df)
        train_df.reset_index(inplace=True, drop=True)
        train_df.reset_index(inplace=True, drop=False)
        test_df = train_test_split.test_df
        make_index_column(test_df)
    else:
        make_index_column(all_data_df)
        train_df = all_data_df
        test_df = all_data_df


    # transform input and output
    column_transformers = IOColumnTransformers(input_column_types, output_column_types, all_data_df)

    # plot data set with and without transformers
    plot_untransformed_and_transformed('train+test labels', all_data_df,
                                       temporal_io_float_column_names.present_columns + non_temporal_float_columns,
                                       input_categorical_columns)

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
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    #scheduler = ReduceLROnPlateau(optimizer, 'min')

    # train and test the model
    first_row: torch.Tensor
    model_output_recording: ModelOutputRecording = ModelOutputRecording(model)

    def train_or_test_SL_epoch(dataloader, model, optimizer, epoch_num, train=True):
        nonlocal first_row, model_output_recording
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        if train:
            model.train()
        else:
            model.eval()
        cumulative_loss = 0
        accuracy = {}
        # bar = Bar('Processing', max=size)
        for name in column_transformers.output_types.column_names():
            accuracy[name] = 0
        with tqdm(total=len(dataloader), disable=False) as pbar:
            for batch, (X, Y) in enumerate(dataloader):
                if batch == 0 and epoch_num == 0 and train:
                    first_row = X[0:1, :]
                X, Y = X.to(device), Y.to(device)
                transformed_Y = column_transformers.transform_columns(False, Y, X)
                # XR = torch.randn_like(X, device=device)
                # XR[:,0] = X[:,0]
                # YZ = torch.zeros_like(Y) + 0.1

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
                if epoch_num == num_epochs - 1:
                    model_output_recording.record_output(pred, Y, transformed_Y, train)
                pbar.update(1)

        cumulative_loss /= num_batches
        for name in column_transformers.output_types.column_names():
            accuracy[name] /= size
        accuracy_string = finish_accuracy(accuracy, column_transformers)
        train_test_str = "Train" if train else "Test"
        print(f"Epoch {train_test_str} Accuracy: {accuracy_string}, Transformed Avg Loss: {cumulative_loss:>8f}")
        return cumulative_loss

    def train_and_test_SL(model, train_dataloader, test_dataloader):
        nonlocal optimizer
        for epoch_num in range(num_epochs):
            print(f"\nEpoch {epoch_num + 1}\n-------------------------------")
            #if epoch_num % 100 == 1000:
                # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            train_loss = train_or_test_SL_epoch(train_dataloader, model, optimizer, epoch_num, True)
            with torch.no_grad():
                train_or_test_SL_epoch(test_dataloader, model, None, epoch_num, False)
            #scheduler.step(train_loss)

    total_train_df = train_df
    train_data = AimDataset(train_df, column_transformers)
    test_data = AimDataset(test_df, column_transformers)
    for dad_num in range(dad_iters + 1):
        print(f"DaD Iter {dad_num + 1}\n-------------------------------")
        # step 1: train model
        # create data sets for pytorch
        total_train_data = AimDataset(total_train_df, column_transformers)

        batch_size = min([64, len(total_train_data), len(test_data)])

        train_dataloader = DataLoader(total_train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        print(f"num train examples: {len(total_train_data)}")
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

        if dad_num < dad_iters:
            # step 2: inference and result collection
            pred_df = on_policy_inference(train_data, train_df, model, column_transformers)
            # model.to(CUDA_DEVICE_STR)

            # step 3: create new training data set
            dad_df = create_dad_dataset(pred_df, train_df)
            total_train_df = pd.concat([total_train_df, dad_df], ignore_index=True)

    model_output_recording.plot(column_transformers, output_column_types.column_names())

    if save:
        script_model = torch.jit.trace(model.to(CPU_DEVICE_STR), first_row)
        script_model.save(Path(__file__).parent / '..' / '..' / 'models' / 'engagement_aim_model' / 'script_model.pt')

    return TrainResult(train_data, test_data, column_transformers, model)


if __name__ == "__main__":
    all_data_df = pd.read_csv(data_path)
    train(all_data_df)
