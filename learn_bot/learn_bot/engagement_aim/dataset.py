import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt

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


    def plot_untransformed_and_transformed(self, cts: IOColumnTransformers, df, transformed_df = None):
        # plot untransformed and transformed outputs
        fig = plt.figure(constrained_layout=True)
        subfigs = fig.subfigures(nrows=2, ncols=1)

        # untransformed
        axs = subfigs[0].subplots(1,2)
        subfigs[0].suptitle('untransformed')
        df.hist('delta view angle x (t - 0)', ax=axs[0], bins=100)
        axs[0].set_xlabel('yaw degree')
        axs[0].set_ylabel('num points')
        df.hist('delta view angle y (t - 0)', ax=axs[1], bins=100)
        axs[1].set_xlabel('pitch degree')

        # transformed
        axs = subfigs[1].subplots(1,2)
        subfigs[1].suptitle('transformed')
        if transformed_df is None:
            transformed_df = pd.DataFrame(
                cts.output_ct.transform(df.loc[:, cts.output_types.column_names()]),
                columns=cts.output_types.column_names())
        transformed_df.hist('delta view angle x (t - 0)', ax=axs[0], bins=100)
        axs[0].set_xlabel('standardized yaw degree')
        axs[0].set_ylabel('num points')
        transformed_df.hist('delta view angle y (t - 0)', ax=axs[1], bins=100)
        axs[1].set_xlabel('standardized pitch degree')
        #plt.tight_layout()
        plt.show()
