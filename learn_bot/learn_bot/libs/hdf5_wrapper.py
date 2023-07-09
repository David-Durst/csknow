from pathlib import Path
from typing import List, Dict, Optional

import h5py
import numpy as np
import pandas as pd

from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.io_transforms import IOColumnTransformers


class HDF5Wrapper:
    hdf5_path: Path
    id_cols: List[str]
    id_df: pd.DataFrame
    sample_df: pd.DataFrame
    input_data: Dict[Path, np.ndarray] = {}
    output_data: Dict[Path, np.ndarray] = {}

    def __init__(self, hdf5_path: Path, id_cols: List[str], id_df: Optional[pd.DataFrame] = None,
                 sample_df: Optional[pd.DataFrame] = None):
        self.hdf5_path = hdf5_path
        self.id_cols = id_cols
        if id_df is None:
            self.id_df = load_hdf5_to_pd(hdf5_path, cols_to_get=id_cols)
        else:
            self.id_df = id_df
        if sample_df is None:
            self.sample_df = load_hdf5_to_pd(hdf5_path, rows_to_get=min(100, len(self.id_df)))
        else:
            self.sample_df = sample_df

    def limit(self, selector_df: pd.Series):
        self.id_df = self.id_df[selector_df]

    def __len__(self):
        return len(self.id_df)

    def clone(self):
        result = HDF5Wrapper(self.hdf5_path, self.id_cols, self.id_df.copy(), self.sample_df.copy())
        return result

    def create_np_array(self, cts: IOColumnTransformers, load_output_data: bool = True):
        # don't need to filder by id_df after limit, as data loader will get the id from the limited id_df for lookup
        # in entire np array
        HDF5Wrapper.input_data[self.hdf5_path] = load_hdf5_to_np_array(self.hdf5_path,
                                                                       cts.input_types.column_names_all_categorical_columns(),
                                                                       True)#[self.id_df.id]
        if load_output_data:
            HDF5Wrapper.output_data[self.hdf5_path] = load_hdf5_to_np_array(self.hdf5_path,
                                                                            cts.output_types.column_names_all_categorical_columns(),
                                                                            False)#[self.id_df.id]

    def get_input_data(self) -> np.ndarray:
        return HDF5Wrapper.input_data[self.hdf5_path]

    def get_output_data(self) -> np.ndarray:
        return HDF5Wrapper.output_data[self.hdf5_path]


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

    def __len__(self):
        return len(self.id_df)

    def clone(self):
        result = PDWrapper(self.hdf5_path, self.sample_df.copy(), self.id_cols)
        return result

    def create_np_array(self, cts: IOColumnTransformers):
        HDF5Wrapper.input_data[self.hdf5_path] = \
            self.sample_df.loc[:, cts.input_types.column_names_all_categorical_columns()].to_numpy(dtype=np.float32)
        HDF5Wrapper.output_data[self.hdf5_path] = \
            self.sample_df.loc[:, cts.output_types.column_names_all_categorical_columns()].to_numpy(dtype=np.float32)

    def get_input_data(self) -> np.ndarray:
        return HDF5Wrapper.input_data[self.hdf5_path]

    def get_output_data(self) -> np.ndarray:
        return HDF5Wrapper.output_data[self.hdf5_path]