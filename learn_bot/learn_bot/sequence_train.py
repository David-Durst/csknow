# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import dataclasses

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pad_packed_sequence, pack_padded_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from pathlib import Path
from learn_bot.baseline import *
from joblib import dump
from learn_bot.sequence_model import SequenceNeuralNetwork, SNNArgs
from learn_bot.sequence_dataset import SequenceBotDatasetArgs, SequenceBotDataset
from dataclasses import dataclass
import matplotlib.pyplot as plt
from learn_bot.sequence_creation import organize_into_sequences

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
    dropped_cols: List[str]
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
        elif col_type == 4:
            columns_by_type.dropped_cols.append(all_cols[col_idx])
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
input_cols_by_type = ColumnsByType([], [], [], [], [], [player_id_col])
set_column_types(2, non_player_id_input_cols, input_cols_by_type)
input_one_hot_cols_nums = compute_one_hot_cols_nums(3)
output_cols = get_config_line(4)
output_cols_by_type = ColumnsByType([], [], [], [], [], [])
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
train_df = all_data_df[all_data_df_split_predicate].copy()
test_df = all_data_df[~all_data_df_split_predicate].copy()

train_sequence_to_elements_df = organize_into_sequences(train_df)
test_sequence_to_elements_df = organize_into_sequences(test_df)

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

fig, axs = plt.subplots(3,2)
plt.suptitle('engine values')
all_data_df.hist('delta view x 1', ax=axs[0,0], bins=100)
all_data_df.hist('delta view y 1', ax=axs[0,1], bins=100)
all_data_df.hist('delta view x 4', ax=axs[1,0], bins=100)
all_data_df.hist('delta view y 4', ax=axs[1,1], bins=100)
all_data_df.hist('delta view x 8', ax=axs[2,0], bins=100)
all_data_df.hist('delta view y 8', ax=axs[2,1], bins=100)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3,2)
plt.suptitle('transformed values')
transformed_output = pd.DataFrame(output_ct.transform(all_data_df.loc[:, output_cols]), columns=output_cols)
transformed_output.hist('delta view x 1', ax=axs[0,0], bins=100)
transformed_output.hist('delta view y 1', ax=axs[0,1], bins=100)
transformed_output.hist('delta view x 4', ax=axs[1,0], bins=100)
transformed_output.hist('delta view y 4', ax=axs[1,1], bins=100)
transformed_output.hist('delta view x 8', ax=axs[2,0], bins=100)
transformed_output.hist('delta view y 8', ax=axs[2,1], bins=100)
plt.tight_layout()
plt.show()

def get_name_range(name: str, ct: ColumnTransformer) -> slice:
    name_indices = [i for i, col_name in enumerate(ct.get_feature_names_out()) if name in col_name]
    if name_indices:
        return range(min(name_indices), max(name_indices) + 1)
    else:
        return range(0,0)
input_ranges = [get_name_range(name, input_ct) for name in input_cols]
output_ranges = [get_name_range(name, output_ct) for name in output_cols]


train_dataset_args = SequenceBotDatasetArgs(input_ct, output_ct, input_cols, output_cols, train_sequence_to_elements_df)
training_data = SequenceBotDataset(train_df, train_dataset_args)
test_dataset_args = SequenceBotDatasetArgs(input_ct, output_ct, input_cols, output_cols, test_sequence_to_elements_df)
test_data = SequenceBotDataset(test_df, test_dataset_args)

batch_size = 16


def pad_collator(batch):
    # https://discuss.pytorch.org/t/why-lengths-should-be-given-in-sorted-order-in-pack-padded-sequence/3540
    # not relevant - https://github.com/pytorch/pytorch/issues/23079
    input_X, input_Y = zip(*batch)
    lens = [len(x) for x in input_X]
    X_padded = pad_sequence(input_X, batch_first=True)
    Y_padded = pad_sequence(input_Y, batch_first=True)
    return X_padded, Y_padded, lens


train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collator)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collator)

for X, Y, lens in train_dataloader:
    print(f"Train shape of X: {X.shape} {X.dtype}")
    print(f"Train shape of Y: {Y.shape} {Y.dtype}")
    print(f"Train lengths: {lens}")
    break

for X, Y, lens in test_dataloader:
    print(f"Test shape of X: {X.shape} {X.dtype}")
    print(f"Test shape of Y: {Y.shape} {Y.dtype}")
    print(f"Test lengths: {lens}")
    break

#baseline_model = BaselineBotModel(training_data.X, training_data.Y, output_names, output_ranges)
#baseline_model.score(test_data.X, test_data.Y)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
embedding_dim = 5
nn_args = SNNArgs(player_id_to_ix, embedding_dim, input_ct, output_ct, output_cols, output_ranges)
model = SequenceNeuralNetwork(nn_args).to(device)
print(model)
params = list(model.parameters())
print("params by layer")
for param_layer in params:
    print(param_layer.shape)

float_loss_fn = nn.MSELoss(reduction='none')
binary_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
classification_loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, y, lens):
    total_loss = 0
    for i in range(len(output_cols)):
        loss_fn = float_loss_fn
        if output_cols[i] in output_cols_by_type.boolean_cols:
            loss_fn = binary_loss_fn
        elif output_cols[i] in output_cols_by_type.categorical_cols:
            loss_fn = classification_loss_fn
        unmasked_loss = loss_fn(pred[:, :, output_ranges[i]], y[:, :, output_ranges[i]])
        mask = torch.zeros_like(unmasked_loss)
        for i, length in enumerate(lens):
            mask[i, 0:length, :] = 1.
        total_loss += torch.sum(unmasked_loss * mask) / sum(lens)
    return total_loss

def get_accuracy_metric_name():
    metric_names = {}
    for name in output_cols:
        if name in output_cols_by_type.boolean_cols:
            metric_names[name] = '0.5 Accuracy'
        elif name in output_cols_by_type.categorical_cols:
            metric_names[name] = 'Category Accuracy'
        else:
            metric_names[name] = 'MSE'
    return metric_names


def compute_accuracy(pred, Y, lens, correct):
    for batch_index, batch_len in enumerate(lens):
        for name, r in zip(output_cols, output_ranges):
            if name in output_cols_by_type.boolean_cols:
                correct[name] += (torch.le(pred[batch_index, 0:batch_len, r], 0.5) == torch.le(Y[batch_index, 0:batch_len, r], 0.5)) \
                    .type(torch.float).sum().item()
            elif name in output_cols_by_type.categorical_cols:
                correct[name] += (pred[batch_index, 0:batch_len, r].argmax(1) == Y[batch_index, 0:batch_len, r].argmax(1)) \
                    .type(torch.float).sum().item()
            else:
                correct[name] += torch.square(pred[batch_index, 0:batch_len, r] -  Y[batch_index, 0:batch_len, r]).sum().item()

first_batch = True
def train_or_test(dataloader, model, optimizer, train = True):
    global first_batch
    size = dataloader.dataset.num_elements()
    num_batches = len(dataloader)
    if train:
        model.train()
    else:
        model.eval()
    cumulative_loss = 0
    correct = {}
    metric_names = get_accuracy_metric_name()
    for name in output_cols:
        correct[name] = 0
    for batch, (X, Y, lens) in enumerate(dataloader):
        # row too big to explore by printing
        #if first_batch and train:
        #    first_batch = False
        #    print(X.cpu().tolist())
        #    print(Y.cpu().tolist())
        X, Y = X.to(device), Y.to(device)
        #XR = torch.randn_like(X)
        #XR[:, :, 0] = X[:, :, 0]
        #YZ = torch.zeros_like(Y) + 0.1

        # Compute prediction error
        pred = model(X, lens)
        batch_loss = compute_loss(pred, Y, lens)
        cumulative_loss += batch_loss

        # Backpropagation
        if train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        if train and batch % 10 == 0:
            loss, current = batch_loss.item(), batch * len(X)
            #print('pred')
            #print(pred[0:2])
            #print('y')
            #print(Y[0:2])
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

        compute_accuracy(pred, Y, lens, correct)
    cumulative_loss /= num_batches
    for name in output_cols:
        correct[name] /= size
    train_test_str = "Train" if train else "Test"
    print(f"Epoch {train_test_str} Avg loss: {cumulative_loss:>8f}, Accuracy Metrics:")
    for name, metric_name in metric_names.items():
        print(f"\t{name} {metric_name}: {correct[name]}")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_or_test(train_dataloader, model, optimizer, True)
    train_or_test(test_dataloader, model, None, False)

dump(train_dataset_args, Path(__file__).parent / '..' / 'model' / 'dataset_args.joblib')
dump(nn_args, Path(__file__).parent / '..' / 'model' / 'nn_args.joblib')
torch.save(model.state_dict(), Path(__file__).parent / '..' / 'model' / 'model.pt')

print("Done")
