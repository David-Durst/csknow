import torch
from torch.utils.data import Dataset
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes, PRIOR_TICKS, FUTURE_TICKS, \
    CUR_TICK
from typing import List
from learn_bot.libs.temporal_column_names import TemporalIOColumnNames


# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class AimDataset(Dataset):
    def __init__(self, df, cts: IOColumnTransformers):
        self.id = df.loc[:, 'id']
        self.round_id = df.loc[:, 'round id']
        self.tick_id = df.loc[:, 'tick id']
        self.engagement_id = df.loc[:, 'engagement id']
        self.attacker_player_id = df.loc[:, 'attacker player id']
        self.victim_player_id = df.loc[:, 'victim player id']
        self.num_shots_fired = df.loc[:, 'num shots fired']
        self.ticks_since_last_fire = df.loc[:, 'last fire tick id']

        round_starts = df.groupby('round id').first('index').loc[:, ['index']].rename(columns={'index': 'start index'})
        round_ends = df.groupby('round id').last('index').loc[:, ['index']].rename(columns={'index': 'end index'})
        self.round_starts_ends = round_starts.join(round_ends, on='round id')

        # convert player id's to indexes
        self.X = torch.tensor(df.loc[:, cts.input_types.column_names()].values).float()
        self.Y = torch.tensor(df.loc[:, cts.output_types.column_names()].values).float()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


base_float_columns: List[str] = ["delta view angle x", "delta view angle y",
                                 "recoil angle x", "recoil angle y",
                                 "delta view angle recoil adjusted x", "delta view angle recoil adjusted y",
                                 "delta position x", "delta position y", "delta position z",
                                 "eye-to-head distance"]

non_temporal_float_columns = ["num shots fired", "ticks since last fire"]

input_categorical_columns: List[str] = ["weapon type"]

temporal_io_float_column_names = TemporalIOColumnNames(base_float_columns, PRIOR_TICKS, CUR_TICK, FUTURE_TICKS)

input_column_types = ColumnTypes(temporal_io_float_column_names.input_columns + non_temporal_float_columns,
                                 input_categorical_columns, [6])

output_column_types = ColumnTypes(temporal_io_float_column_names.output_columns, [], [])
