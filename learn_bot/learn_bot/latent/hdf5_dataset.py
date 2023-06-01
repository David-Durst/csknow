import pandas as pd
import torch
from torch.utils.data import Dataset

from learn_bot.libs.hdf5_to_pd import HDF5Wrapper
from learn_bot.libs.io_transforms import IOColumnTransformers
from pathlib import Path
from learn_bot.latent.order.column_names import *

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class LatentDataset(Dataset):
    x_cols: List[str]
    y_cols: List[str]

    def __init__(self, cts: IOColumnTransformers, ):
        self.x_cols = cts.input_types.column_names_all_categorical_columns()
        self.y_cols = cts.output_types.column_names_all_categorical_columns()

    def open_hdf5(self):
        self.hdf5 = HDF5Wrapper()
        self.dataset = self.img_hdf5['dataset']  # if you want dataset.

    def __getitem__(self, item: int):
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
        img0 = self.img_hdf5['dataset'][0]  # Do loading here
        img1 = self.dataset[1]
        return img0, img1

    def __len__(self):
        if self.windowed:
            return int(len(self.id) / 10)
        else:
            return len(self.id)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
