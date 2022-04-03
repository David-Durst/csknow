# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
from learn_bot.baseline import *
from joblib import dump
from learn_bot.model import NeuralNetwork, NNArgs
from learn_bot.dataset import BotDataset, BotDatasetArgs

all_data_df = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'engagement' / 'train_engagement_dataset.csv')

# load config
def get_config_line(i):
    if config_lines[i] == '\n':
        return []
    else:
        return config_lines[i].strip().split(',')

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
input_one_hot_cols = get_config_line(2)
input_one_hot_cols_nums = compute_one_hot_cols_nums(3)
input_min_max_cols = get_config_line(4)
output_cols = get_config_line(5)
output_one_hot_cols = get_config_line(6)
output_one_hot_cols_nums = compute_one_hot_cols_nums(7)
output_min_max_cols = get_config_line(8)
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
passthrough_input_cols = compute_passthrough_cols(input_cols, input_one_hot_cols, input_min_max_cols)
if passthrough_input_cols:
    input_transformers.append(('pass', 'passthrough', passthrough_input_cols))
if input_one_hot_cols:
    input_transformers.append(('one-hot', OneHotEncoder(categories=input_one_hot_cols_nums), input_one_hot_cols))
if input_min_max_cols:
    input_transformers.append(('zero-to-one', MinMaxScaler(), input_min_max_cols))
input_ct = ColumnTransformer(transformers=input_transformers, sparse_threshold=0)

output_transformers = []
passthrough_output_cols = compute_passthrough_cols(output_cols, output_one_hot_cols, output_min_max_cols)
if passthrough_output_cols:
    output_transformers.append(('pass', 'passthrough', passthrough_output_cols))
if output_one_hot_cols:
    output_transformers.append(('one-hot', OneHotEncoder(categories=output_one_hot_cols_nums), output_one_hot_cols))
if output_min_max_cols:
    output_transformers.append(('zero-to-one', MinMaxScaler(), output_min_max_cols))
output_ct = ColumnTransformer(transformers=output_transformers, sparse_threshold=0)
# remember: fit Y is ignored for this fitting as not SL
input_ct.fit(all_data_df.loc[:, input_cols])
output_ct.fit(all_data_df.loc[:, output_cols])

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

float_loss_fn = nn.L1Loss()
binary_loss_fn = nn.BCEWithLogitsLoss()
classification_loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, y):
    total_loss = 0
    for i in range(len(output_cols)):
        loss_fn = binary_loss_fn
        if output_cols[i] in passthrough_output_cols:
            loss_fn = float_loss_fn
        elif output_cols[i] in output_one_hot_cols:
            loss_fn = classification_loss_fn
        total_loss += loss_fn(pred[:, output_ranges[i]], y[:, output_ranges[i]])
    return total_loss

def compute_accuracy(pred, Y, correct):
    for name, r in zip(output_cols, output_ranges):
        if name in passthrough_output_cols:
            correct[name] += torch.square(pred[:, r] -  Y[:, r]).sum().item()
        elif name in output_one_hot_cols:
            correct[name] += (pred[:, r].argmax(1) == Y[:, r].argmax(1)) \
                .type(torch.float).sum().item()
        else:
            correct[name] += (torch.le(pred[:, r], 0.5) == torch.le(Y[:, r], 0.5)) \
                .type(torch.float).sum().item()

first_batch = True
def train(dataloader, model, optimizer):
    global first_batch
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    correct = {}
    for name in output_cols:
        correct[name] = 0
    for batch, (X, Y) in enumerate(dataloader):
        if first_batch:
            first_batch = False
            print(X.cpu().tolist())
            print(Y.cpu().tolist())
        X, Y = X.to(device), Y.to(device)

        # Compute prediction error
        pred = model(X)
        batch_loss = compute_loss(pred, Y)
        train_loss += batch_loss

        # Backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = batch_loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        compute_accuracy(pred, Y, correct)
    train_loss /= num_batches
    for name in output_cols:
        correct[name] /= size
        correct[name] *= 100
    print(f"Epoch Train Error: Accuracy: {correct}%, Avg loss: {train_loss:>8f} \n")


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = {}
    for name in output_cols:
        correct[name] = 0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            test_loss += compute_loss(pred, Y).item()
            compute_accuracy(pred, Y, correct)
    test_loss /= num_batches
    for name in output_cols:
        correct[name] /= size
        correct[name] *= 100
    print(f"Epoch Test Error: Accuracy: {correct}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, optimizer)
    test(test_dataloader, model)

dump(dataset_args, Path(__file__).parent / '..' / 'model' / 'dataset_args.joblib')
dump(nn_args, Path(__file__).parent / '..' / 'model' / 'nn_args.joblib')
torch.save(model.state_dict(), Path(__file__).parent / '..' / 'model' / 'model.pt')

print("Done")
