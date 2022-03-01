# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from learn_bot.baseline import *

all_data_df = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'train_dataset.csv')

# load config
def get_config_line(i):
    return config_lines[i].strip().split(',')


config_file = open(Path(__file__).parent / '..' / 'data' / 'train_config.csv', 'r')
config_lines = config_file.readlines()
config_file.close()
player_id_col = get_config_line(0)[0]
non_player_id_input_cols = get_config_line(1)
input_cols = [player_id_col] + non_player_id_input_cols
input_one_hot_cols = get_config_line(2)
input_min_max_cols = get_config_line(3)
output_cols = get_config_line(4)
output_one_hot_cols = get_config_line(5)
output_min_max_cols = get_config_line(6)
all_data_cols = input_cols + output_cols

# for good embeddings, make sure player indices are from 0 to max-1, makes sure missing ids aren't problem
unique_player_id = all_data_df.loc[:, player_id_col].unique()
player_id_to_ix = {player_id: i for (i, player_id) in enumerate(unique_player_id)}
all_data_df = all_data_df.replace({player_id_col: player_id_to_ix})

train_df, test_df = train_test_split(all_data_df, test_size=0.2)


def compute_passthrough_cols(all_cols, *non_passthrough_lists):
    non_passthrough_cols = []
    for non_passthrough_list in non_passthrough_lists:
        non_passthrough_cols += non_passthrough_list
    return [c for c in all_cols if c not in non_passthrough_cols]

# transform concatenates outputs in order provided, so this ensures that source player id comes first
# as that is only pass through col
input_ct = ColumnTransformer(transformers=[
    ('pass', 'passthrough', compute_passthrough_cols(input_cols, input_one_hot_cols, input_min_max_cols)),
    ('one-hot', OneHotEncoder(), input_one_hot_cols),
    ('zero-to-one', MinMaxScaler(), input_min_max_cols),
])
output_ct = ColumnTransformer(transformers=[
    ('pass', 'passthrough', compute_passthrough_cols(output_cols, output_one_hot_cols, output_min_max_cols)),
    ('one-hot', OneHotEncoder(), output_one_hot_cols),
    ('zero-to-one', MinMaxScaler(), output_min_max_cols),
])
# remember: fit Y is ignored for this fitting as not SL
input_ct.fit(all_data_df.loc[:, input_cols])
output_ct.fit(all_data_df.loc[:, output_cols])

def get_name_range(name: str) -> slice:
    name_indices = [i for i, col_name in enumerate(output_ct.get_feature_names_out()) if name in col_name]
    return slice(min(name_indices), max(name_indices) + 1)
output_names = ['nav target', 'shoot', 'crouch']
output_ranges = [get_name_range(name) for name in output_names]

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class BotDataset(Dataset):
    def __init__(self, df):
        self.id = df.iloc[:, 0]
        self.tick_id = df.iloc[:, 1]
        self.source_player_id = df.loc[:, player_id_col]
        self.source_player_name = df.iloc[:, 3]
        self.demo_name = df.iloc[:, 4]

        # convert player id's to indexes
        self.X = torch.tensor(input_ct.transform(df.loc[:, input_cols])).float()
        #Y_prescale_df = df.loc[:, output_cols]
        #target_cols = [column for column in df.columns if column.startswith('nav') and column.endswith('target')]
        #num_target_cols = len(target_cols)
        #Y_scaled_df = min_max_scaler.transform(Y_prescale_df)
        #df['moving'] = np.where((df['delta x']**2 + df['delta y']**2) ** 0.5 > 0.5, 1.0, 0.0)
        #df['not moving'] = np.where((df['delta x']**2 + df['delta y']**2) ** 0.5 > 0.5, 0.0, 1.0)
        #sub_df = Y_prescale_df[target_cols + ['shoot next true', 'shoot next false',
        #             'crouch next true', 'crouch next false']].values
        self.Y = torch.tensor(output_ct.transform(df.loc[:, output_cols])).float()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


training_data = BotDataset(train_df)
test_data = BotDataset(test_df)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

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

embedding_dim = 5
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.embeddings = nn.Embedding(len(unique_player_id), embedding_dim)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embedding_dim + input_ct.get_feature_names_out().size - 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.moveLayer = nn.Sequential(
            nn.Linear(128, output_ranges[0].stop - output_ranges[0].start),
        )
        self.crouchLayer = nn.Sequential(
            nn.Linear(128, 2),
        )
        self.shootLayer = nn.Sequential(
            nn.Linear(128, 2),
        )
        #self.moveSigmoid = nn.Sigmoid()

    def forward(self, x):
        idx, x_vals = x.split([1, x.shape[1] - 1], dim=1)
        idx_long = idx.long()
        embeds = self.embeddings(idx_long).view((-1, embedding_dim))
        x_all = torch.cat((embeds, x_vals), 1)
        logits = self.linear_relu_stack(x_all)
        moveOutput = self.moveLayer(logits)
        #moveOutput = self.moveSigmoid(self.moveLayer(logits))
        crouchOutput = self.crouchLayer(logits)
        shootOutput = self.shootLayer(logits)
        return torch.cat((moveOutput, crouchOutput, shootOutput), dim=1)


model = NeuralNetwork().to(device)
print(model)
params = list(model.parameters())
print("params by layer")
for param_layer in params:
    print(param_layer.shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, y):
    move_loss = loss_fn(pred[:, output_ranges[0]], y[:, output_ranges[0]])
    shoot_loss = loss_fn(pred[:, output_ranges[1]], y[:, output_ranges[1]])
    crouch_loss = loss_fn(pred[:, output_ranges[2]], y[:, output_ranges[2]])
    return move_loss + shoot_loss + crouch_loss


def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        # Compute prediction error
        pred = model(X)
        total_loss = compute_loss(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = total_loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = {}
    for name in output_names:
        correct[name] = 0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            test_loss += compute_loss(pred, Y).item()
            for name, r in zip(output_names, output_ranges):
                correct[name] += (pred[:, r].argmax(1) == Y[:, r].argmax(1)) \
                    .type(torch.float).sum().item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    for name in output_names:
        correct[name] /= size
        correct[name] *= 100
    print(f"Test Error: \n Accuracy: {correct}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, optimizer)
    test(test_dataloader, model)

#baseline_model = BaselineBotModel(training_data.X, training_data.Y, prediction_names,
#                                  prediction_range_starts, prediction_range_ends)
#baseline_model.score(test_data.X, test_data.Y)

print("Done")