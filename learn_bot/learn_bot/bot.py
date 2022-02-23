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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pathlib import Path

all_data_df = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'train_dataset.csv')
train_df, test_df = train_test_split(all_data_df, test_size=0.2)
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(all_data_df.iloc[:, 95:].values)

unique_player_id = all_data_df.iloc[:,2].unique()
player_id_to_ix = {player_id: i for (i, player_id) in enumerate(unique_player_id)}
embedding_dim = 5


# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class BotDataset(Dataset):
    def __init__(self, df):
        self.id = df.iloc[:, 0]
        self.tick_id = df.iloc[:, 1]
        self.source_player_id = df.iloc[:, 2]
        self.source_player_name = df.iloc[:, 3]
        self.demo_name = df.iloc[:, 4]
        x_cols = [all_data_df.columns[2]] + all_data_df.columns[5:94].tolist()
        # convert player id's to indexes
        df_with_ixs = df.replace({all_data_df.columns[2]: player_id_to_ix})
        self.X = torch.tensor(df_with_ixs.loc[:, x_cols].values).float()
        Y_prescale_df = df.iloc[:, 95:].values
        Y_scaled_df = min_max_scaler.transform(Y_prescale_df)
        self.Y = torch.tensor(Y_scaled_df).float()

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
            nn.Linear(89 + embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.moveLayer = nn.Sequential(
            nn.Linear(128, 2),
            nn.ReLU(),
        )
        self.crouchLayer = nn.Sequential(
            nn.Linear(128, 2),
            nn.ReLU(),
        )
        self.shootLayer = nn.Sequential(
            nn.Linear(128, 2),
            nn.ReLU(),
        )
        self.moveSigmoid = nn.Sigmoid()
        self.crouchSigmoid = nn.Softmax(dim=1)
        self.shootSigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        idx, x_vals = torch.split(x, [1, 89], dim=1)
        idx_long = idx.long()
        embeds = self.embeddings(idx_long).view((-1, embedding_dim))
        x_all = torch.cat((embeds, x_vals), 1)
        logits = self.linear_relu_stack(x_all)
        moveOutput = self.moveSigmoid(self.moveLayer(logits))
        crouchOutput = self.crouchSigmoid(self.crouchLayer(logits))
        shootOutput = self.shootSigmoid(self.shootLayer(logits))
        return torch.cat((moveOutput, crouchOutput, shootOutput), dim=1)


model = NeuralNetwork().to(device)
print(model)
params = list(model.parameters())
print("params by layer")
for param_layer in params:
    print(param_layer.shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, Y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    #correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done")