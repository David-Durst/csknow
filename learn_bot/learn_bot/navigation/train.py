import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from learn_bot.engagement_aim.column_management import CUDA_DEVICE_STR, CPU_DEVICE_STR
from learn_bot.libs.accuracy_and_loss import finish_accuracy, compute_loss, compute_accuracy
from learn_bot.libs.df_grouping import train_test_split_by_col
from learn_bot.libs.temporal_column_names import TemporalIOColumnNames
from learn_bot.navigation.cnn_nav_model import CNNNavModel
from learn_bot.navigation.dataset import NavDataset
from learn_bot.navigation.io_transforms import PRIOR_TICKS, CUR_TICK, FUTURE_TICKS, ColumnTypes, \
    IOColumnAndImageTransformers
from typing import List
import time


start_time = time.perf_counter()

csv_outputs_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs'

non_img_df = pd.read_csv(csv_outputs_path / 'trainNav.csv')

non_img_df = non_img_df.head(n=5000)

train_test_split = train_test_split_by_col(non_img_df, 'trajectory id')
train_df = train_test_split.train_df
test_df = train_test_split.test_df

# create all training data column names
# little clunky as the column generator assumes that t>=0 is label and all t=0 are to be vis'd
base_prior_img_columns: List[str] = ["player pos", "player vis", "player vis from",
                                     "distance map", "friendly pos", "friendly vis", "friendly vis from",
                                     "vis enemies", "vis from enemies", "c4 pos"]
prior_temporal_img_columns = TemporalIOColumnNames(base_prior_img_columns, PRIOR_TICKS, 0, 0)

base_future_img_columns: List[str] = ["goal pos"]
future_temporal_img_columns = TemporalIOColumnNames(base_prior_img_columns, PRIOR_TICKS, 0, 0)

# looking into the future for goal region (higher level model will give us this)
temporal_img_column_names = TemporalIOColumnNames([], 0, 0, 0)
temporal_img_column_names.input_columns = prior_temporal_img_columns.input_columns + \
                                          future_temporal_img_columns.output_columns
temporal_img_column_names.vis_columns = temporal_img_column_names.input_columns
temporal_img_column_names.output_columns = []

base_prior_float_columns: List[str] = ["player view dir x", "player view dir y", "health", "armor"]
prior_temporal_float_columns = TemporalIOColumnNames(base_prior_float_columns, PRIOR_TICKS, 0, 0)

cur_cat_columns: List[str] = ["movement result x", "movement result y"]

# transform input and output
input_column_types = ColumnTypes(prior_temporal_float_columns.input_columns, [], [])

output_column_types = ColumnTypes([], cur_cat_columns, [3, 3])

# weird dance as column transformers need data set to compute mean/std dev
# but data set needs column transformers during train/inference time
# so make data set without transformers, then use dataset during transformer creation, then pass
# transformers to dataset
train_nav_dataset = NavDataset(train_df, csv_outputs_path / 'trainNavData', temporal_img_column_names.vis_columns)
column_transformers = IOColumnAndImageTransformers(input_column_types, output_column_types, train_df,
                                                   train_nav_dataset.get_img_sample())
train_nav_dataset.add_column_transformers(column_transformers)

test_nav_dataset = NavDataset(test_df, csv_outputs_path / 'trainNavData', temporal_img_column_names.vis_columns)
test_nav_dataset.add_column_transformers(column_transformers)

batch_size = 64

train_dataloader = DataLoader(train_nav_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_nav_dataset, batch_size=batch_size, shuffle=True)

for non_img_X, img_X, Y in train_dataloader:
    print(f"Train shape of non_img_X: {non_img_X.shape} {non_img_X.dtype}")
    print(f"Train shape of img_X: {img_X.shape} {img_X.dtype}")
    print(f"Train shape of Y: {Y.shape} {Y.dtype}")
    break

for non_img_X, img_X, Y in test_dataloader:
    print(f"Test shape of non_img_X: {non_img_X.shape} {non_img_X.dtype}")
    print(f"Test shape of img_X: {img_X.shape} {img_X.dtype}")
    print(f"Test shape of Y: {Y.shape} {Y.dtype}")
    break

# Get cpu or gpu device for training.
device: str = CUDA_DEVICE_STR if torch.cuda.is_available() else CPU_DEVICE_STR
print(f"Using {device} device")

# Define model
embedding_dim = 5
model = CNNNavModel(column_transformers).to(device)
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
first_batch = True
first_row_non_img: torch.Tensor
first_row_img: torch.Tensor
def train_or_test(dataloader, model, optimizer, epoch_num, train = True):
    global first_batch, first_row_non_img, first_row_img, model_output_recording
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
    for batch, (non_img_X, img_X, Y) in enumerate(dataloader):
        if batch == 15:
            exit(0)
        if first_batch and train:
            first_batch = False
            #print(X.cpu().tolist())
            #print(Y.cpu().tolist())
            first_row_non_img = non_img_X[0:1,:]
            first_row_img = img_X[0:1, :]
        non_img_X, img_X, Y = non_img_X.to(device), img_X.to(device), Y.to(device)
        transformed_Y = column_transformers.transform_columns(False, Y)
        #XR = torch.randn_like(X, device=device)
        #XR[:,0] = X[:,0]
        #YZ = torch.zeros_like(Y) + 0.1

        # Compute prediction error
        pred = model(non_img_X, img_X)
        batch_loss = compute_loss(pred, transformed_Y, column_transformers)
        cumulative_loss += batch_loss

        # Backpropagation
        if train:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        compute_accuracy(pred, Y, accuracy, column_transformers)

    cumulative_loss /= num_batches
    for name in column_transformers.output_types.column_names():
        accuracy[name] /= size
    accuracy_string = finish_accuracy(accuracy, column_transformers)
    train_test_str = "Train" if train else "Test"
    print(f"Epoch {train_test_str} Accuracy: {accuracy_string}, Transformed Avg Loss: {cumulative_loss:>8f}")


epochs = 10
for epoch_num in range(epochs):
    print(f"\nEpoch {epoch_num+1}\n-------------------------------")
    train_or_test(train_dataloader, model, optimizer, epoch_num, True)
    train_or_test(test_dataloader, model, None, epoch_num, False)

script_model = torch.jit.trace(model.to(CPU_DEVICE_STR), (first_row_non_img, first_row_img))
script_model.save(Path(__file__).parent / '..' / '..' / 'models' / 'nav_model' / 'script_model.pt')

print("Done")
