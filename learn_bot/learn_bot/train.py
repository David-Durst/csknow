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
from learn_bot.baseline import *
from joblib import dump
from learn_bot.model import NeuralNetwork, NNArgs
from learn_bot.dataset import BotDataset, BotDatasetArgs
from dataclasses import dataclass
import matplotlib.pyplot as plt

all_data_df = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'engagement' / 'train_engagement_dataset.csv')


# load config
def get_config_line(i):
    if config_lines[i] == '\n':
        return []
    else:
        return config_lines[i].strip().split(',')


@dataclass
class ColumnsByType:
    boolean_cols: List[str]
    float_min_max_cols: List[str]
    float_non_linear_cols: List[str]
    categorical_cols: List[str]
    bookkeeping_passthrough_cols: List[str]


def set_column_types(config_line_num, all_cols, columns_by_type: ColumnsByType):
    col_types_strs = get_config_line(config_line_num)
    for col_idx in range(len(col_types_strs)):
        col_type = int(col_types_strs[col_idx])
        col_str = all_cols[col_idx]
        if col_type == 0:
            columns_by_type.boolean_cols.append(all_cols[col_idx])
        elif col_type == 1:
            columns_by_type.float_min_max_cols.append(all_cols[col_idx])
        elif col_type == 2:
            columns_by_type.float_non_linear_cols.append(all_cols[col_idx])
        elif col_type == 3:
            columns_by_type.categorical_cols.append(all_cols[col_idx])
        else:
            print(f'''invalid col type {col_types_strs[col_idx]}''')


def compute_one_hot_cols_nums(config_row_num):
    result = []
    if config_lines[config_row_num].strip() == "":
        return result
    one_hot_cols_nums = config_lines[config_row_num].strip().split(',')
    for one_hot_col_nums in one_hot_cols_nums:
        # if this is a list of nums, take the list directly
        if one_hot_col_nums[0] == "\"":
            one_hot_col_nums_no_quotes = one_hot_col_nums[1:-1]
            result.append(sorted([int(num) for num in one_hot_col_nums_no_quotes.split(';')]))
        # if this a single num, expand it from from 0 to num-1
        else:
            result.append(list(range(int(one_hot_col_nums))))
    return result


config_file = open(Path(__file__).parent / '..' / 'data' / 'engagement' / 'train_config.csv', 'r')
config_lines = config_file.readlines()
config_file.close()
player_id_col = get_config_line(0)[0]
non_player_id_input_cols = get_config_line(1)
input_cols = [player_id_col] + non_player_id_input_cols
input_cols_by_type = ColumnsByType([], [], [], [], [player_id_col])
set_column_types(2, non_player_id_input_cols, input_cols_by_type)
input_one_hot_cols_nums = compute_one_hot_cols_nums(3)
output_cols = get_config_line(4)
output_cols_by_type = ColumnsByType([], [], [], [], [])
set_column_types(5, output_cols, output_cols_by_type)
output_one_hot_cols_nums = compute_one_hot_cols_nums(6)
all_data_cols = input_cols + output_cols

# for good embeddings, make sure player indices are from 0 to max-1, makes sure missing ids aren't problem
unique_player_id = all_data_df.loc[:, player_id_col].unique()
player_id_to_ix = {player_id: i for (i, player_id) in enumerate(unique_player_id)}
all_data_df = all_data_df.replace({player_id_col: player_id_to_ix})
#all_data_df = all_data_df[all_data_df['team'] == 0]

# train test split on rounds with rounds weighted by number of entries in each round
# so 80-20 train test split on actual data with rounds kept coherent
# split by rounds, weight rounds by number of values in each round
per_round_df = all_data_df.groupby(['round id']).count()
# sample frac = 1 to shuffle
random_sum_rounds_df = per_round_df.sample(frac=1).cumsum()
top_80_pct_rounds = random_sum_rounds_df[random_sum_rounds_df['id'] < 0.8 * len(all_data_df)].index.to_list()
all_data_df_split_predicate = all_data_df['round id'].isin(top_80_pct_rounds)
train_df = all_data_df[all_data_df_split_predicate]
test_df = all_data_df[~all_data_df_split_predicate]


def compute_passthrough_cols(all_cols, *non_passthrough_lists):
    non_passthrough_cols = []
    for non_passthrough_list in non_passthrough_lists:
        non_passthrough_cols += non_passthrough_list
    return [c for c in all_cols if c not in non_passthrough_cols]

# transform concatenates outputs in order provided, so this ensures that source player id comes first
# as that is only pass through col
input_transformers = []
output_transformers = []


# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
def create_column_transformers(transformers, cols_by_type: ColumnsByType, one_hot_cols_nums):
    if cols_by_type.boolean_cols or cols_by_type.bookkeeping_passthrough_cols:
        transformers.append(('pass', 'passthrough', cols_by_type.bookkeeping_passthrough_cols + cols_by_type.boolean_cols))
    if cols_by_type.categorical_cols:
        transformers.append(
            ('one-hot', OneHotEncoder(categories=one_hot_cols_nums), cols_by_type.categorical_cols))
    if cols_by_type.float_min_max_cols:
        transformers.append(('zero-to-one-min-max', MinMaxScaler(), cols_by_type.float_min_max_cols))
    if cols_by_type.float_non_linear_cols:
        transformers.append(('zero-to-one-non-linear', QuantileTransformer(output_distribution='normal'), cols_by_type.float_non_linear_cols))


create_column_transformers(input_transformers, input_cols_by_type, input_one_hot_cols_nums)
input_ct = ColumnTransformer(transformers=input_transformers, sparse_threshold=0)
create_column_transformers(output_transformers, output_cols_by_type, output_one_hot_cols_nums)
output_ct = ColumnTransformer(transformers=output_transformers, sparse_threshold=0)

# remember: fit Y is ignored for this fitting as not SL
input_ct.fit(train_df.loc[:, input_cols])
output_ct.fit(train_df.loc[:, output_cols])

def get_name_range(name: str) -> slice:
    name_indices = [i for i, col_name in enumerate(output_ct.get_feature_names_out()) if name in col_name]
    if name_indices:
        return range(min(name_indices), max(name_indices) + 1)
    else:
        return range(0,0)
output_ranges = [get_name_range(name) for name in output_cols]


dataset_args = BotDatasetArgs(input_ct, output_ct, input_cols, output_cols)
training_data = BotDataset(train_df, dataset_args)
test_data = BotDataset(test_df, dataset_args)

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

#baseline_model = BaselineBotModel(training_data.X, training_data.Y, output_names, output_ranges)
#baseline_model.score(test_data.X, test_data.Y)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
embedding_dim = 5
nn_args = NNArgs(player_id_to_ix, embedding_dim, input_ct, output_ct, output_cols, output_ranges)
model = NeuralNetwork(nn_args).to(device)
print(model)
params = list(model.parameters())
print("params by layer")
for param_layer in params:
    print(param_layer.shape)

float_loss_fn = nn.MSELoss()
binary_loss_fn = nn.BCEWithLogitsLoss()
classification_loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, y):
    total_loss = 0
    for i in range(len(output_cols)):
        loss_fn = float_loss_fn
        if output_cols[i] in output_cols_by_type.boolean_cols:
            loss_fn = binary_loss_fn
        elif output_cols[i] in output_cols_by_type.categorical_cols:
            loss_fn = classification_loss_fn
        total_loss += loss_fn(pred[:, output_ranges[i]], y[:, output_ranges[i]])
    return total_loss

def compute_accuracy(pred, Y, correct):
    for name, r in zip(output_cols, output_ranges):
        if name in output_cols_by_type.boolean_cols:
            correct[name] += (torch.le(pred[:, r], 0.5) == torch.le(Y[:, r], 0.5)) \
                .type(torch.float).sum().item()
        elif name in output_cols_by_type.categorical_cols:
            correct[name] += (pred[:, r].argmax(1) == Y[:, r].argmax(1)) \
                .type(torch.float).sum().item()
        else:
            correct[name] += torch.square(pred[:, r] -  Y[:, r]).sum().item()

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

dump(dataset_args, Path(__file__).parent / '..' / 'model' / 'dataset_args.joblib')
dump(nn_args, Path(__file__).parent / '..' / 'model' / 'nn_args.joblib')
torch.save(model.state_dict(), Path(__file__).parent / '..' / 'model' / 'model.pt')

print("Done")
