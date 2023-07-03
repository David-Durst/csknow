import pandas as pd
import torch
from torch.utils.data import Dataset
from learn_bot.libs.io_transforms import IOColumnTransformers
from pathlib import Path
from learn_bot.latent.engagement.column_names import *

latent_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'behaviorTreeFeatureStore.hdf5'
latent_window_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'behaviorTreeWindowFeatureStore.hdf5'

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class LatentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cts: IOColumnTransformers):
        # convert player id's to indexes
        self.X = torch.tensor(df.loc[:, cts.input_types.column_names_all_categorical_columns()].to_numpy()).float()
        self.Y = torch.tensor(df.loc[:, cts.output_types.column_names_all_categorical_columns()].to_numpy()).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
