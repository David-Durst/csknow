# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import dataclasses

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from dataset import *
from joblib import dump
from dataclasses import dataclass
import matplotlib.pyplot as plt
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes
from learn_bot.engagement_aim.linear_model import LinearModel
from learn_bot.engagement_aim.output_plotting import plot_untransformed_and_transformed, ModelOutputRecording
from typing import Dict, List
from math import sqrt

all_data_df = pd.read_csv(Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'engagementAim.csv')

# train test split on rounds with rounds weighted by number of entries in each round
# so 80-20 train test split on actual data with rounds kept coherent
# split by rounds, weight rounds by number of values in each round
per_round_df = all_data_df.groupby(['engagement id']).count()
# sample frac = 1 to shuffle
random_sum_rounds_df = per_round_df.sample(frac=1).cumsum()
# top 80% of engagements (summing by ticks per engagement to weight by ticks) are training data, rest are test
top_80_pct_rounds = random_sum_rounds_df[random_sum_rounds_df['id'] < 0.8 * len(all_data_df)].index.to_list()
all_data_df_split_predicate = all_data_df['engagement id'].isin(top_80_pct_rounds)
train_df = all_data_df[all_data_df_split_predicate]
test_df = all_data_df[~all_data_df_split_predicate]

base_float_columns: List[str] = ["delta view angle x ", "delta view angle y ",
                                 "recoil angle x ", "recoil angle y ",
                                 "delta view angle recoil adjusted x ", "delta view angle recoil adjusted y ",
                                 "eye-to-eye distance "]

input_float_columns: List[str] = []
output_float_columns: List[str] = []
vis_float_columns: List[str] = []
PRIOR_TICKS = -12
FUTURE_TICKS = 6
CUR_TICK = 1
for i in range(PRIOR_TICKS, FUTURE_TICKS+CUR_TICK):
    offset_str = "(t"
    if i < 0:
        offset_str += str(i)
    elif i > 0:
        offset_str += "+" + str(i)
    offset_str += ")"

    if i < 0:
        for base_col in base_float_columns:
            input_float_columns.append(base_col + offset_str)
    else:
        for base_col in base_float_columns:
            if i == 0:
                vis_float_columns.append(base_col + offset_str)
            output_float_columns.append(base_col + offset_str)

input_categorical_columns: List[str] = ["weapon type"]
vis_categorical_columns: List[str] = input_categorical_columns

# transform input and output
input_column_types = ColumnTypes(input_float_columns, [], [], input_categorical_columns, [6], [], [])

output_column_types = ColumnTypes(output_float_columns, [], [], [], [], [], [])

column_transformers = IOColumnTransformers(input_column_types, output_column_types, all_data_df)

vis_float_cols_transformed_df = pd.DataFrame(
    column_transformers.output_ct.transform(all_data_df.loc[:, output_column_types.column_names()]),
    columns=output_column_types.column_names())

# plot data set with and without transformers
plot_untransformed_and_transformed('train+test labels', column_transformers, all_data_df, vis_float_columns,
                                   input_categorical_columns, vis_float_cols_transformed_df)

# create data sets for pytorch
training_data = AimDataset(train_df, column_transformers)
test_data = AimDataset(test_df, column_transformers)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, Y in train_dataloader:
    print(f"Train shape of X: {X.shape} {X.dtype}")
    print(f"Train shape of Y: {Y.shape} {Y.dtype}")
    break

for X, Y in test_dataloader:
    print(f"Test shape of X: {X.shape} {X.dtype}")
    print(f"Test shape of Y: {Y.shape} {Y.dtype}")
    break

# Get cpu or gpu device for training.
device: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
embedding_dim = 5
model = LinearModel(column_transformers).to(device)
print(model)
params = list(model.parameters())
print("params by layer")
for param_layer in params:
    print(param_layer.shape)

# define losses
float_loss_fn = nn.MSELoss()
binary_loss_fn = nn.BCEWithLogitsLoss()
classification_loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

output_cols = column_transformers.output_types.column_names()
output_ranges = column_transformers.get_name_ranges(False)


# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, y):
    total_loss = 0
    for i in range(len(output_cols)):
        loss_fn = float_loss_fn
        if output_cols[i] in column_transformers.output_types.boolean_cols:
            loss_fn = binary_loss_fn
        elif output_cols[i] in column_transformers.output_types.categorical_cols:
            loss_fn = classification_loss_fn
        total_loss += loss_fn(pred[:, output_ranges[i]], y[:, output_ranges[i]])
    return total_loss


def compute_accuracy(pred, Y, accuracy):
    for name, r in zip(output_cols, output_ranges):
        # compute accuracy using unnormalized outputs on end
        untransformed_r = range(r.start + len(output_cols), r.stop + len(output_cols))
        if name in column_transformers.output_types.float_standard_cols or \
                name in column_transformers.output_types.float_min_max_cols or \
                name in column_transformers.output_types.float_non_linear_cols:
            accuracy[name] += torch.square(pred[:, untransformed_r] - Y[:, r]).sum().item()
        elif name in column_transformers.output_types.boolean_cols:
            accuracy[name] += (torch.le(pred[:, untransformed_r], 0.5) == torch.le(Y[:, r], 0.5)) \
                .type(torch.float).sum().item()
        elif name in column_transformers.output_types.categorical_cols:
            accuracy[name] += (pred[:, untransformed_r].argmax(1) == Y[:, r].argmax(1)) \
                .type(torch.float).sum().item()
        else:
            raise "Invalid Column Type For compute_accuracy"


def finish_accuracy(accuracy):
    accuracy_string = ""
    for name, unadjusted_r in zip(output_cols, output_ranges):
        # make float accuracy into rmse
        if name in column_transformers.output_types.float_standard_cols or \
                name in column_transformers.output_types.float_min_max_cols or \
                name in column_transformers.output_types.float_non_linear_cols:
            accuracy[name] = sqrt(accuracy[name])
            accuracy_string += f'''{name}: {accuracy[name]} rmse'''
        # record top-1 accuracy for tohers
        elif name in column_transformers.output_types.boolean_cols:
            accuracy_string += f'''{name}: {accuracy[name]} bool eq'''
        elif name in column_transformers.output_types.categorical_cols:
            accuracy_string += f'''{name}: {accuracy[name]} % cat top 1 acc'''
        else:
            raise "Invalid Column Type For finish_accuracy"
        accuracy_string += "; "
    return accuracy_string

# train and test the model
first_batch = True
first_row: torch.Tensor
model_output_recording: ModelOutputRecording = ModelOutputRecording(column_transformers)
def train_or_test(dataloader, model, optimizer, train = True):
    global first_batch, first_row, model_output_recording
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if train:
        model.train()
    else:
        model.eval()
    cumulative_loss = 0
    accuracy = {}
    for name in output_cols:
        accuracy[name] = 0
    for batch, (X, Y) in enumerate(dataloader):
        if first_batch and train:
            first_batch = False
            #print(X.cpu().tolist())
            #print(Y.cpu().tolist())
            first_row = X[0:1,:]
        X, Y = X.to(device), Y.to(device)
        transformed_Y = model.convert_output_labels_for_loss(Y)
        #XR = torch.randn_like(X, device=device)
        #XR[:,0] = X[:,0]
        #YZ = torch.zeros_like(Y) + 0.1

        # Compute prediction error
        pred = model(X)
        batch_loss = compute_loss(pred, transformed_Y)
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

        compute_accuracy(pred, Y, accuracy)
        #model_output_recording.record_output(column_transformers, pred, Y, transformed_Y, train)

    cumulative_loss /= num_batches
    for name in output_cols:
        accuracy[name] /= size
    accuracy_string = finish_accuracy(accuracy)
    train_test_str = "Train" if train else "Test"
    print(f"Epoch {train_test_str} Accuracy: {accuracy_string}, Transformed Avg Loss: {cumulative_loss:>8f}")


epochs = 5
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train_or_test(train_dataloader, model, optimizer, True)
    train_or_test(test_dataloader, model, None, False)

model_output_recording.plot(column_transformers, output_float_columns)

dump(column_transformers, Path(__file__).parent / '..' / '..' / 'models' / 'engagement_aim_model' /
     'column_transformers.joblib')
torch.save(model.state_dict(), Path(__file__).parent / '..' / '..' / 'models' / 'engagement_aim_model' / 'model.pt')
cpu_model = model.to('cpu')
script_model = torch.jit.trace(cpu_model, first_row)
script_model.save(Path(__file__).parent / '..' / '..' / 'models' / 'engagement_aim_model' / 'script_model.pt')

print("Done")
