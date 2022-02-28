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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pathlib import Path
from learn_bot.baseline import *

all_data_df = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'train_dataset.csv')
config = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'train_config.csv')
train_df, test_df = train_test_split(all_data_df, test_size=0.2)
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(all_data_df.iloc[:, config.at[0, 'label main min']:].values)

unique_player_id = all_data_df.iloc[:,2].unique()
player_id_to_ix = {player_id: i for (i, player_id) in enumerate(unique_player_id)}
embedding_dim = 5

non_embedding_features = 0
num_target_cols = 0
# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class BotDataset(Dataset):
    def __init__(self, df):
        global non_embedding_features, num_target_cols
        self.id = df.iloc[:, 0]
        self.tick_id = df.iloc[:, 1]
        self.source_player_id = df.iloc[:, config.at[0,'data player id']]
        self.source_player_name = df.iloc[:, 3]
        self.demo_name = df.iloc[:, 4]

        x_cols = [all_data_df.columns[config.at[0,'data player id']]] + \
                 all_data_df.columns[config.at[0,'data main min']:config.at[0,'data main max']].tolist()
        non_embedding_features = len(x_cols) - 1
        # convert player id's to indexes
        df_with_ixs = df.replace({all_data_df.columns[2]: player_id_to_ix})
        self.X = torch.tensor(df_with_ixs.loc[:, x_cols].values).float()
        Y_prescale_df = df.iloc[:, config.at[0, 'label main min']:]
        target_cols = [column for column in df.columns if column.startswith('nav') and column.endswith('target')]
        num_target_cols = len(target_cols)
        #Y_scaled_df = min_max_scaler.transform(Y_prescale_df)
        #df['moving'] = np.where((df['delta x']**2 + df['delta y']**2) ** 0.5 > 0.5, 1.0, 0.0)
        #df['not moving'] = np.where((df['delta x']**2 + df['delta y']**2) ** 0.5 > 0.5, 0.0, 1.0)
        sub_df = Y_prescale_df[target_cols + ['shoot next true', 'shoot next false',
                     'crouch next true', 'crouch next false']].values
        self.Y = torch.tensor(sub_df).float()

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

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.embeddings = nn.Embedding(len(unique_player_id), embedding_dim)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(non_embedding_features + embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.moveLayer = nn.Sequential(
            nn.Linear(128, num_target_cols),
        )
        self.crouchLayer = nn.Sequential(
            nn.Linear(128, 2),
        )
        self.shootLayer = nn.Sequential(
            nn.Linear(128, 2),
        )
        #self.moveSigmoid = nn.Sigmoid()

    def forward(self, x):
        idx, x_vals = torch.split(x, [1, non_embedding_features], dim=1)
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

prediction_names = ['move nav area', 'shoot', 'crouch']
prediction_range_starts = [0, num_target_cols, num_target_cols+2]
prediction_range_ends = [num_target_cols, num_target_cols+2, num_target_cols+4]
# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, y):
    move_loss = loss_fn(pred[:, prediction_range_starts[0]:prediction_range_ends[0]],
                        y[:, prediction_range_starts[0]:prediction_range_ends[0]])
    shoot_loss = loss_fn(pred[:, prediction_range_starts[1]:prediction_range_ends[1]],
                        y[:, prediction_range_starts[1]:prediction_range_ends[1]])
    crouch_loss = loss_fn(pred[:, prediction_range_starts[2]:prediction_range_ends[2]],
                        y[:, prediction_range_starts[2]:prediction_range_ends[2]])
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
    for name in prediction_names:
        correct[name] = 0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            test_loss += compute_loss(pred, Y).item()
            for i, name in enumerate(prediction_names):
                correct[name] += (pred[:, prediction_range_starts[i]:prediction_range_ends[i]].argmax(1) ==
                                Y[:, prediction_range_starts[i]:prediction_range_ends[i]].argmax(1)) \
                    .type(torch.float).sum().item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    for name in prediction_names:
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