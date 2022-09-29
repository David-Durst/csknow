import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from typing import List, Optional

from learn_bot.engagement_aim.column_management import IOColumnTransformers


# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class AimDataset(Dataset):
    def __init__(self, df, cts: IOColumnTransformers):
        self.id = df.loc[:, 'id']
        self.tick_id = df.loc[:, 'tick id']
        self.round_id = df.loc[:, 'engagement id']
        self.attacker_player_id = df.loc[:, 'attacker player id']
        self.victim_player_id = df.loc[:, 'victim player id']

        # convert player id's to indexes
        self.X = torch.tensor(df.loc[:, cts.input_types.column_names()].values).float()
        self.Y = torch.tensor(df.loc[:, cts.output_types.column_names()].values).float()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
