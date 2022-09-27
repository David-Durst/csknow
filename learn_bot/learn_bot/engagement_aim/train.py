# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import dataclasses

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from pathlib import Path
from dataset import *
from joblib import dump
from dataclasses import dataclass
import matplotlib.pyplot as plt
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes
from learn_bot.engagement_aim.linear_model import LinearModel

all_data_df = pd.read_csv(Path(__file__).parent / '..' / '..' / 'data' / 'engagementAim.csv')

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


# transform input and output
input_column_types = ColumnTypes(["delta view angle x (t - 1)", "delta view angle y (t - 1)", "eye-to-eye distance (t - 1)",
                                  "delta view angle x (t - 2)", "delta view angle y (t - 2)", "eye-to-eye distance (t - 2)",
                                  "delta view angle x (t - 3)", "delta view angle y (t - 3)", "eye-to-eye distance (t - 3)",
                                  "delta view angle x (t - 4)", "delta view angle y (t - 4)", "eye-to-eye distance (t - 4)",
                                  "delta view angle x (t - 5)", "delta view angle y (t - 5)", "eye-to-eye distance (t - 5)"])

output_column_types = ColumnTypes(["delta view angle x (t - 0)","delta view angle y (t - 0)"])

column_transformers = IOColumnTransformers(input_column_types, output_column_types, all_data_df)

# plot transformed output so it looks reasonable
fig, axs = plt.subplots(1,2)
transformed_output = pd.DataFrame(
    column_transformers.output_ct.transform(all_data_df.loc[:, output_column_types.get_all_columns()]),
    columns=output_column_types.get_all_columns())
transformed_output.hist('delta view angle x (t - 0)', ax=axs[0, 0], bins=100)
transformed_output.hist('delta view angle x (t - 0)', ax=axs[0, 1], bins=100)
plt.tight_layout()
plt.show()

# create data sets
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
device = "cuda" if torch.cuda.is_available() else "cpu"
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

output_cols = column_transformers.output_types.get_all_columns()
output_ranges = column_transformers.get_output_name_ranges()


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


def compute_accuracy(pred, Y, correct):
    for name, r in zip(output_cols, output_ranges):
        if name in column_transformers.output_types.boolean_cols:
            correct[name] += (torch.le(pred[:, r], 0.5) == torch.le(Y[:, r], 0.5)) \
                .type(torch.float).sum().item()
        elif name in column_transformers.output_types.categorical_cols:
            correct[name] += (pred[:, r].argmax(1) == Y[:, r].argmax(1)) \
                .type(torch.float).sum().item()
        else:
            correct[name] += torch.square(pred[:, r] -  Y[:, r]).sum().item()


# train and test the model
first_batch = True
def train_or_test(dataloader, model, optimizer, train = True):
    global first_batch
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if train:
        model.train()
    else:
        model.eval()
    cumulative_loss = 0
    correct = {}
    for name in output_cols:
        correct[name] = 0
    for batch, (X, Y) in enumerate(dataloader):
        if first_batch and train:
            first_batch = False
            print(X.cpu().tolist())
            print(Y.cpu().tolist())
        X, Y = X.to(device), Y.to(device)
        #XR = torch.randn_like(X, device=device)
        #XR[:,0] = X[:,0]
        #YZ = torch.zeros_like(Y) + 0.1

        # Compute prediction error
        pred = model(X)
        batch_loss = compute_loss(pred, Y)
        cumulative_loss += batch_loss

        # Backpropagation
        if train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        if train and batch % 100 == 0:
            loss, current = batch_loss.item(), batch * len(X)
            print('pred')
            print(pred[0:2])
            print('y')
            print(Y[0:2])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        compute_accuracy(pred, Y, correct)
    cumulative_loss /= num_batches
    for name in output_cols:
        correct[name] /= size
    train_test_str = "Train" if train else "Test"
    print(f"Epoch {train_test_str} Error: Accuracy: {correct}%, Avg loss: {cumulative_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_or_test(train_dataloader, model, optimizer, True)
    train_or_test(train_dataloader, model, None, False)

dump(column_transformers, Path(__file__).parent / '..' / '..' / 'models' / 'engagement_aim_model' /
     'column_transformers.joblib')
torch.save(model.state_dict(), Path(__file__).parent / '..' / '..' / 'models' / 'engagement_aim_model' / 'model.pt')

print("Done")
