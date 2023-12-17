import os
from pathlib import Path
from typing import List, Dict, Optional

import h5py
import numpy as np
import pandas as pd
import torch

from learn_bot.libs.df_grouping import make_index_column
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.io_transforms import IOColumnTransformers


class HDF5Wrapper:
    hdf5_path: Path
    id_cols: List[str]
    id_df: pd.DataFrame
    sample_df: pd.DataFrame
    vis_cols: Optional[List[str]]
    vis_np: Optional[np.ndarray]
    input_data: Dict[Path, np.ndarray] = {}
    output_data: Dict[Path, np.ndarray] = {}

    def __init__(self, hdf5_path: Path, id_cols: List[str], id_df: Optional[pd.DataFrame] = None,
                 sample_df: Optional[pd.DataFrame] = None, vis_cols: Optional[List[str]] = None):
        self.hdf5_path = hdf5_path
        self.id_cols = id_cols
        if id_df is None:
            self.id_df = load_hdf5_to_pd(hdf5_path, cols_to_get=id_cols)
            # hack to make sure that ints are preserved in usual case
            if 'test success' in self.id_df:
                self.id_df = self.id_df.astype({'test success': 'object'})
            make_index_column(self.id_df)
        else:
            self.id_df = id_df
        if sample_df is None:
            self.sample_df = load_hdf5_to_pd(hdf5_path, rows_to_get=min(100, len(self.id_df)))
        else:
            self.sample_df = sample_df
        self.vis_cols = vis_cols
        self.vis_np = None

    def limit(self, selector_df: pd.Series):
        self.id_df = self.id_df[selector_df]
        make_index_column(self.id_df)

    def add_extra_column(self, extra_column_name: str, extra_column: pd.Series):
        self.id_df[extra_column_name] = extra_column

    def __len__(self):
        return len(self.id_df)

    def clone(self):
        result = HDF5Wrapper(self.hdf5_path, self.id_cols, self.id_df.copy(), self.sample_df.copy())
        result.vis_cols = self.vis_cols
        result.vis_np = self.vis_np
        return result

    def create_np_array(self, cts: IOColumnTransformers, load_output_data: bool = True):
        # don't need to filter by id_df after limit, as data loader will get the id from the limited id_df for lookup
        # in entire np array
        input_mmap_path = (self.hdf5_path.parent / ('input_mmap_' + self.hdf5_path.name)).with_suffix('.npy')
        output_mmap_path = (self.hdf5_path.parent / ('output_mmap_' + self.hdf5_path.name)).with_suffix('.npy')

        hdf5_timestamp = os.path.getmtime(self.hdf5_path)
        input_timestamp = os.path.getmtime(input_mmap_path) if input_mmap_path.is_file() else None
        output_timestamp = os.path.getmtime(output_mmap_path) if output_mmap_path.is_file() else None

        # no need to recreate np files if they were made after hdf5, as hdf5 haven't updated
        if input_timestamp is None or input_timestamp < hdf5_timestamp:
            input_np = load_hdf5_to_np_array(self.hdf5_path, cts.input_types.column_names_all_categorical_columns(),
                                             True).astype('float16')
            np.save(str(input_mmap_path), input_np, allow_pickle=False)
        HDF5Wrapper.input_data[self.hdf5_path] = np.load(str(input_mmap_path), mmap_mode='r')
        if load_output_data:
            if output_timestamp is None or output_timestamp < hdf5_timestamp:
                output_np = load_hdf5_to_np_array(self.hdf5_path,
                                                  cts.output_types.column_names_all_categorical_columns(),
                                                  False).astype('float16')
                np.save(str(output_mmap_path), output_np, allow_pickle=False)
            HDF5Wrapper.output_data[self.hdf5_path] = np.load(str(output_mmap_path), mmap_mode='r')

    def get_input_data(self, index: int) -> np.ndarray:
        return HDF5Wrapper.input_data[self.hdf5_path][index]

    def get_all_input_data(self) -> np.ndarray:
        # last indexing necessary to apply id_df based test/train limits
        return HDF5Wrapper.input_data[self.hdf5_path][self.id_df['id']]

    def get_output_data(self, index: int) -> np.ndarray:
        return HDF5Wrapper.output_data[self.hdf5_path][index]

    def get_all_output_data(self) -> np.ndarray:
        return HDF5Wrapper.output_data[self.hdf5_path][self.id_df['id']]

    def get_vis_data(self) -> np.ndarray:
        if self.vis_cols is None:
            raise Exception("calling get_vis_data without loading vis data")

        vis_mmap_path = (self.hdf5_path.parent / ('vis_mmap_' + self.hdf5_path.name)).with_suffix('.npy')

        hdf5_timestamp = os.path.getmtime(self.hdf5_path)
        vis_timestamp = os.path.getmtime(vis_mmap_path) if vis_mmap_path.is_file() else None

        if vis_timestamp is None or vis_timestamp < hdf5_timestamp:
            vis_np = load_hdf5_to_np_array(self.hdf5_path, cols_to_get=self.vis_cols, cast_to_float=True)
            np.save(str(vis_mmap_path), vis_np, allow_pickle=False)

        if self.vis_np is None:
            self.vis_np = np.load(str(vis_mmap_path), mmap_mode='r')
        return self.vis_np[self.id_df['id']]

    def get_extra_df(self, cols: List[str]) -> pd.DataFrame:
        return load_hdf5_to_pd(self.hdf5_path, cols_to_get=cols, root_key='extra')


def load_hdf5_to_np_array(hdf5_path: Path, cols_to_get: List[str], cast_to_float: bool) -> np.ndarray:
    # get data as numpy arrays and column names
    #np_arrs: List[np.ndarray] = []
    #col_names: List[List[str]] = []
    result = None
    with h5py.File(hdf5_path) as hdf5_file:
        hdf5_data = hdf5_file['data']
        for i, k in enumerate(cols_to_get):
            np_arr: np.ndarray = hdf5_data[k][...]
            if cast_to_float:
                np_arr = np_arr.astype(np.float32)
            if i == 0:
                result = np.empty([len(cols_to_get), len(np_arr)], dtype=np_arr.dtype)
            result[i] = np_arr
    return result.transpose()


class PDWrapper(HDF5Wrapper):
    def __init__(self, hdf5_path: Path, df: pd.DataFrame, id_cols: List[str]):
        self.hdf5_path = hdf5_path
        self.id_cols = id_cols
        self.id_df = df.loc[:, id_cols]
        self.sample_df = df

    def limit(self, selector_df: pd.Series):
        self.id_df = self.id_df[selector_df]

    def add_extra_column(self, extra_column_name: str, extra_column: pd.Series):
        self.id_df[extra_column_name] = extra_column

    def __len__(self):
        return len(self.id_df)

    def clone(self):
        result = PDWrapper(self.hdf5_path, self.sample_df.copy(), self.id_cols)
        return result

    def create_np_array(self, cts: IOColumnTransformers, load_output_data=True):
        HDF5Wrapper.input_data[self.hdf5_path] = \
            self.sample_df.loc[:, cts.input_types.column_names_all_categorical_columns()].to_numpy(dtype=np.float32)
        HDF5Wrapper.output_data[self.hdf5_path] = \
            self.sample_df.loc[:, cts.output_types.column_names_all_categorical_columns()].to_numpy(dtype=np.float32)

    def get_input_data(self) -> np.ndarray:
        return HDF5Wrapper.input_data[self.hdf5_path]

    def get_output_data(self) -> np.ndarray:
        return HDF5Wrapper.output_data[self.hdf5_path]
