import pandas as pd
import torch
from torch.utils.data import Dataset

from learn_bot.latent.place_area.column_names import get_similarity_column
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper
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
        hdf5_id = self.data_hdf5.id_df.iloc[idx].loc['id']
        x_tensor = torch.tensor(self.data_hdf5.get_input_data(hdf5_id))
        y_tensor = torch.tensor(self.data_hdf5.get_output_data(hdf5_id))
        return x_tensor, y_tensor

tmp = 0

class MultipleLatentHDF5Dataset(Dataset):
    x_cols: List[str]
    y_cols: List[str]
    data_hdf5s: List[HDF5Wrapper]
    hdf5s_starts: List[int]
    hdf5s_cum_len: List[int]
    total_len: int
    duplicate_last_equal_to_rest: bool

    def __init__(self, data_hdf5s: List[HDF5Wrapper], cts: IOColumnTransformers,
                 duplicate_last_equal_to_rest: bool = False):
        self.x_cols = cts.input_types.column_names_all_categorical_columns()
        self.y_cols = cts.output_types.column_names_all_categorical_columns()
        self.data_hdf5s = data_hdf5s
        self.total_len = 0
        self.hdf5s_starts = []
        self.hdf5s_cum_len = []
        self.duplicate_last_equal_to_rest = duplicate_last_equal_to_rest
        for i, data_hdf5 in enumerate(data_hdf5s):
            self.hdf5s_starts.append(self.total_len)
            if not duplicate_last_equal_to_rest or i < len(data_hdf5s) - 1:
                self.total_len += len(data_hdf5)
                self.hdf5s_cum_len.append(self.total_len)
            else:
                # handle cases where prior hdf5 have less elements than one to duplicate
                self.total_len += max(self.total_len // 4, len(data_hdf5))
                self.hdf5s_cum_len.append(self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        x_tensor0, y_tensor0, similarity_tensor0, duplicated_last0, indices0 = self.inner_getitem(idx)
        x_tensor1, y_tensor1, similarity_tensor1, duplicated_last1, indices1 = \
            self.inner_getitem(min(idx+1, len(self) - 1))
        return torch.stack([x_tensor0, x_tensor1]), \
            torch.stack([y_tensor0, y_tensor1]), \
            torch.stack([similarity_tensor0, similarity_tensor1]), \
            torch.stack([duplicated_last0, duplicated_last1]), \
            torch.stack([indices0, indices1])

    def inner_getitem(self, idx):
        # hdf5_index selects file, idx_in_hdf5 selects location in file
        hdf5_index = 0
        idx_in_hdf5 = 0
        for i, cum_len in enumerate(self.hdf5s_cum_len):
            if idx < cum_len:
                hdf5_index = i
                idx_in_hdf5 = idx - self.hdf5s_starts[i]
                # this ensures wrap around when duplicating last
                idx_in_hdf5 %= len(self.data_hdf5s[i])
                break
        id_series = self.data_hdf5s[hdf5_index].id_df.iloc[idx_in_hdf5]
        if get_similarity_column(0) in id_series.index:
            similarity_tensor = torch.tensor([id_series.loc[get_similarity_column(0)], id_series.loc[get_similarity_column(1)]])
        else:
            # for now, only thing without hdf5 is bot data, which always pushes to objective
            similarity_tensor = torch.tensor([True, True])
        hdf5_id = id_series.loc['id']
        x_tensor = torch.tensor(self.data_hdf5s[hdf5_index].get_input_data(hdf5_id))
        y_tensor = torch.tensor(self.data_hdf5s[hdf5_index].get_output_data(hdf5_id))
        #x_tensor = torch.tensor(self.data_hdf5s[hdf5_index].input_data[self.data_hdf5s[hdf5_index].hdf5_path][hdf5_id])
        #y_tensor = torch.tensor(self.data_hdf5s[hdf5_index].input_data[self.data_hdf5s[hdf5_index].hdf5_path][hdf5_id])
        return x_tensor, y_tensor, similarity_tensor, \
            torch.tensor(self.duplicate_last_equal_to_rest and (hdf5_index == len(self.data_hdf5s) - 1)), \
            torch.tensor([idx, hdf5_index, hdf5_id])
