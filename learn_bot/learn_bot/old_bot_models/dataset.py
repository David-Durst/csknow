import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class BotDatasetArgs:
    input_ct: ColumnTransformer
    output_ct: ColumnTransformer
    input_cols: List[str]
    output_cols: List[str]


# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class BotDataset(Dataset):
    def __init__(self, df, args: BotDatasetArgs, skip_strings=False):
        self.id = df.loc[:, 'id']
        self.tick_id = df.loc[:, 'tick id']
        self.round_id = df.loc[:, 'round id']
        self.source_player_id = df.loc[:, 'source player id']
        if not skip_strings:
            self.source_player_name = df.loc[:, 'source player name']
            self.demo_name = df.loc[:, 'demo name']
        self.team = df.loc[:, 'team']

        # convert player id's to indexes
        self.X = torch.tensor(args.input_ct.transform(df.loc[:, args.input_cols])).float()
        self.Y = torch.tensor(args.output_ct.transform(df.loc[:, args.output_cols])).float()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
