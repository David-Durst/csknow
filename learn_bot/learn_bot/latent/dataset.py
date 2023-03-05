import pandas as pd
import torch
from torch.utils.data import Dataset
from learn_bot.libs.io_transforms import IOColumnTransformers
from pathlib import Path
from learn_bot.latent.column_names import *

data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'engagementAim.csv'
hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'engagementAim.hdf5'
latent_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'latentEngagementAim.hdf5'
manual_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'manual_outputs' / 'engagementAim.csv'

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class LatentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cts: IOColumnTransformers):
        self.id = df.loc[:, row_id_column]
        self.round_id = df.loc[:, round_id_column]
        self.tick_id = df.loc[:, tick_id_column]
        self.player_id = df.loc[:, player_id_column]

        round_starts = df.groupby(round_id_column).first('index').loc[:, ['index']].rename(columns={'index': 'start index'})
        round_ends = df.groupby(round_id_column).last('index').loc[:, ['index']].rename(columns={'index': 'end index'})
        self.round_starts_ends = round_starts.join(round_ends, on=round_id_column)

        # convert player id's to indexes
        self.X = torch.tensor(df.loc[:, cts.input_types.column_names()].to_numpy()).float()
        self.Y = torch.tensor(df.loc[:, cts.output_types.column_names()].to_numpy()).float()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
