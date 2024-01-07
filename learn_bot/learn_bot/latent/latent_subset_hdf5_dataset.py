from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from learn_bot.latent.place_area.column_names import get_base_similarity_column, get_tick_similarity_column
from learn_bot.libs.io_transforms import IOColumnTransformers, column_names_to_index_list
from pathlib import Path
from learn_bot.latent.engagement.column_names import *

latent_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'behaviorTreeFeatureStore.hdf5'
latent_window_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'behaviorTreeWindowFeatureStore.hdf5'

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class LatentSubsetHDF5Dataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, id_df: pd.DataFrame):
        # convert player id's to indexes
        self.X = X
        self.Y = Y
        self.similarity_columns = [get_tick_similarity_column(0)]
        self.similarity_tensor = torch.tensor(id_df.loc[:, self.similarity_columns].to_numpy()).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.similarity_tensor[idx]
