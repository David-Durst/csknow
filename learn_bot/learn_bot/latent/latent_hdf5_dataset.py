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
    hdf5_path: str
    id_cols: List[str]
    hdf5_wrapper: HDF5Wrapper

    def __init__(self, cts: IOColumnTransformers, data_hdf5: HDF5Wrapper):
        self.x_cols = cts.input_types.column_names_all_categorical_columns()
        self.y_cols = cts.output_types.column_names_all_categorical_columns()
        self.hdf5_path = data_hdf5.hdf5_path
        self.id_cols = data_hdf5.id_cols

    def open_hdf5(self):
        self.hdf5_wrapper = HDF5Wrapper(self.hdf5_path, self.id_cols)

    def __len__(self):
        if not hasattr(self, 'hdf5_wrapper'):
            self.open_hdf5()
        return len(self.hdf5_wrapper)

    def __getitem__(self, idx):
        if not hasattr(self, 'hdf5_wrapper'):
            self.open_hdf5()
        x_tensor = torch.zeros([len(self.x_cols)])
        y_tensor = torch.zeros([len(self.y_cols)])
        hdf5_id = self.hdf5_wrapper.id_df[idx]
        for i in range(len(self.x_cols)):
            x_tensor[i] = self.hdf5_wrapper.hdf5_data[self.x_cols[i]][hdf5_id]
        for i in range(len(self.y_cols)):
            y_tensor[i] = self.hdf5_wrapper.hdf5_data[self.y_cols[i]][hdf5_id]
        return x_tensor, y_tensor
