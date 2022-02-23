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
from pathlib import Path

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class BotDataset(Dataset):
    def __init__(self, file):
        all_data = pd.read_csv(file)
        self.id = all_data.iloc[:,0]
        self.tick_id = all_data.iloc[:,1]
        self.source_player_name = all_data.iloc[:,2]
        self.source_player_id = all_data.iloc[:,3]
        self.demo_name = all_data.iloc[:,4]
        self.X = torch.tensor(all_data.iloc[:,5:94].values).float()
        Y_prescale_df = all_data.iloc[:, 95:].values
        min_max_scaler = preprocessing.MinMaxScaler()
        Y_scaled_df = min_max_scaler.fit_transform(Y_prescale_df)
        self.Y = torch.tensor(Y_scaled_df).float()
        print(self.Y.dtype)
        print("hi")


    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


training_data = BotDataset(Path(__file__).parent / '..' / 'data' / 'train_dataset.csv')

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

for X, Y in train_dataloader:
    print(f"Shape of X: {X.shape} {X.dtype}")
    print(f"Shape of Y: {Y.shape} {Y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(89, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.moveLayer = nn.Sequential(
            nn.Linear(512, 2),
            nn.ReLU(),
        )
        self.crouchLayer = nn.Sequential(
            nn.Linear(512, 2),
            nn.ReLU(),
        )
        self.shootLayer = nn.Sequential(
            nn.Linear(512, 2),
            nn.ReLU(),
        )
        self.moveSigmoid = nn.Sigmoid()
        self.crouchSigmoid = nn.Softmax(dim=1)
        self.shootSigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        moveOutput = self.moveSigmoid(self.moveLayer(logits))
        crouchOutput = self.crouchSigmoid(self.crouchLayer(logits))
        shootOutput = self.shootSigmoid(self.shootLayer(logits))
        return torch.cat((moveOutput, crouchOutput, shootOutput), dim=1)


model = NeuralNetwork().to(device)
print(model)

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

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    #test(test_dataloader, model, loss_fn)
print("Done!")