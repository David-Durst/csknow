import pandas as pd
import torch
from torch.utils.data import Dataset
from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, ColumnTypes, PRIOR_TICKS, FUTURE_TICKS, \
    CUR_TICK, DeltaColumn
from typing import List
from learn_bot.libs.temporal_column_names import TemporalIOColumnNames, get_temporal_field_str
from pathlib import Path
from column_names import *

data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'engagementAim.csv'

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class AimDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cts: IOColumnTransformers):
        self.id = df.loc[:, 'id']
        self.round_id = df.loc[:, 'round id']
        self.tick_id = df.loc[:, 'tick id']
        self.engagement_id = df.loc[:, 'engagement id']
        self.attacker_player_id = df.loc[:, 'attacker player id']
        self.victim_player_id = df.loc[:, 'victim player id']

        round_starts = df.groupby('round id').first('index').loc[:, ['index']].rename(columns={'index': 'start index'})
        round_ends = df.groupby('round id').last('index').loc[:, ['index']].rename(columns={'index': 'end index'})
        self.round_starts_ends = round_starts.join(round_ends, on='round id')

        # convert player id's to indexes
        self.X = torch.tensor(df.loc[:, cts.input_types.column_names()].to_numpy()).float()
        self.Y = torch.tensor(df.loc[:, cts.output_types.column_names()].to_numpy()).float()
        self.Targets = torch.tensor(df.loc[:, cts.output_types.delta_float_target_column_names()].to_numpy()).float()
        self.attacking = torch.tensor(df.loc[:, temporal_io_attacking_column_names.present_columns +
                                                temporal_io_attacking_column_names.future_columns].to_numpy()).float()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Targets[idx], self.attacking[idx]


seconds_per_tick = 1. / 128. * 1000.
