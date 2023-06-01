import pandas as pd
import torch
from torch.utils.data import Dataset

from learn_bot.libs.hdf5_to_pd import HDF5Wrapper
from learn_bot.libs.io_transforms import IOColumnTransformers
from pathlib import Path
from learn_bot.latent.order.column_names import *

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class LatentHDF5Dataset(Dataset):
    x_cols: List[str]
    y_cols: List[str]
    data_hdf5: HDF5Wrapper

    def __init__(self, data_hdf5: HDF5Wrapper, cts: IOColumnTransformers):
        self.x_cols = cts.input_types.column_names_all_categorical_columns()
        self.y_cols = cts.output_types.column_names_all_categorical_columns()
        self.data_hdf5 = data_hdf5

    def __len__(self):
        return len(self.data_hdf5)

    def __getitem__(self, idx):
        x_tensor = torch.zeros([len(self.x_cols)])
        y_tensor = torch.zeros([len(self.y_cols)])
        hdf5_id = self.data_hdf5.id_df.iloc[idx].loc['id']
        for i in range(len(self.x_cols)):
            x_tensor[i] = float(self.data_hdf5.get_data()[self.x_cols[i]][hdf5_id])
        for i in range(len(self.y_cols)):
            y_tensor[i] = float(self.data_hdf5.get_data()[self.y_cols[i]][hdf5_id])
        return x_tensor, y_tensor
