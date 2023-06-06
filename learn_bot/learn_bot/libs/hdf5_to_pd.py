from tempfile import NamedTemporaryFile

import h5py
import numpy
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Set, Optional, Dict

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
            self.sample_df = load_hdf5_to_pd(hdf5_path, rows_to_get=[i for i in range(min(100, len(self.id_df)))])
        else:
            self.sample_df = sample_df

    def limit(self, selector_df: pd.Series):
        self.id_df = self.id_df[selector_df]

    def __len__(self):
        return len(self.id_df)

    def clone(self):
        result = HDF5Wrapper(self.hdf5_path, self.id_cols, self.id_df.copy(), self.sample_df.copy())
        return result

    def create_np_array(self, cts: IOColumnTransformers):
        HDF5Wrapper.input_data[self.hdf5_path] = load_hdf5_to_np_array(self.hdf5_path,
                                                                       cts.input_types.column_names_all_categorical_columns())
        HDF5Wrapper.output_data[self.hdf5_path] = load_hdf5_to_np_array(self.hdf5_path,
                                                                        cts.output_types.column_names_all_categorical_columns())

    def get_input_data(self) -> np.ndarray:
        return HDF5Wrapper.input_data[self.hdf5_path]

    def get_output_data(self) -> np.ndarray:
        return HDF5Wrapper.output_data[self.hdf5_path]



def load_hdf5_to_pd(hdf5_path: Path, selector_df: Optional[pd.DataFrame] = None, cols_to_get: Optional[List] = None,
                    rows_to_get: Optional[List[int]] = None):
    # get data as numpy arrays and column names
    #np_arrs: List[np.ndarray] = []
    #col_names: List[List[str]] = []
    partial_dfs: List[pd.DataFrame] = []
    with h5py.File(hdf5_path) as hdf5_file:
        hdf5_data = hdf5_file['data']
        for k in hdf5_data.keys():
            if cols_to_get is not None and k not in cols_to_get:
                continue
            if rows_to_get is None:
                np_arr: np.ndarray = hdf5_data[k][...]
            else:
                np_arr: np.ndarray = hdf5_data[k][rows_to_get]
            col_names: List[str]
            if hdf5_data[k].attrs:
                col_names = hdf5_data[k].attrs['column names'].split(',')
            else:
                col_names = [k]

            if len(np_arr.shape) == 2 and np_arr.shape[1] > np_arr.shape[0]:
                df_to_append = pd.DataFrame(np_arr.transpose(), columns=col_names)
                partial_dfs.append()
            else:
                df_to_append = pd.DataFrame(np_arr, columns=col_names)

            if selector_df is not None:
                df_to_append = df_to_append.loc[selector_df]

            partial_dfs.append(df_to_append)

    result_df = pd.concat(partial_dfs, axis=1)
    for col in result_df.columns:
        if str(result_df[col].dtype) == 'bool':
            result_df[col] = result_df[col].astype(int)
    return result_df


def compare_to_csv(new_df: pd.DataFrame, old_df: pd.DataFrame):
    assert(sorted(new_df.columns) == sorted(old_df.columns))

    for col_name in new_df.columns:
        if new_df[col_name].dtype.name != 'bool':
            if not np.allclose(np.around(new_df[col_name], 6), np.around(old_df[col_name], 6)):
                is_close = np.isclose(np.around(new_df[col_name], 6), np.around(old_df[col_name], 6))
                first_fail = list(is_close).index(False)
                print(f'''{col_name} isn't close, ''' +
                      f'''example new: {new_df[col_name][first_fail]} ; old: {old_df[col_name][first_fail]}''')
        else:
            if not np.array_equal(new_df[col_name], old_df[col_name]):
                is_equal = np.equal(new_df[col_name], old_df[col_name])
                first_fail = list(is_equal).index(False)
                print(f'''{col_name} isn't close, ''' +
                      f'''example new: {new_df[col_name][first_fail]} ; old: {old_df[col_name][first_fail]}''')


def load_hdf5_extra_to_list(hdf5_path: Path) -> List[List[int]]:
    result: List[List[int]] = []
    with h5py.File(hdf5_path) as hdf5_file:
        hdf5_data = hdf5_file['extra']
        for k in hdf5_data.keys():
            np_arr: np.ndarray = hdf5_data[k][...]
            result.append(np_arr.tolist())
    return result


def load_hdf5_to_np_array(hdf5_path: Path, cols_to_get: List[str]) -> np.ndarray:
    # get data as numpy arrays and column names
    #np_arrs: List[np.ndarray] = []
    #col_names: List[List[str]] = []
    result = None
    with h5py.File(hdf5_path) as hdf5_file:
        hdf5_data = hdf5_file['data']
        for i, k in enumerate(cols_to_get):
            np_arr: np.ndarray = hdf5_data[k][...]
            np_arr = np_arr.astype(np.float32)
            if i == 0:
                result = np.empty([len(cols_to_get), len(np_arr)], dtype=np.float32)
            result[i] = np_arr
    return result.transpose()


class PDWrapper(HDF5Wrapper):
    def __init__(self, df: pd.DataFrame, id_cols: List[str]):
        self.hdf5_path = None
        self.id_cols = id_cols
        self.id_df = df.loc[:, id_cols]
        self.sample_df = df

    def limit(self, selector_df: pd.Series):
        self.id_df = self.id_df[selector_df]

    def __len__(self):
        return len(self.id_df)

    def clone(self):
        result = PDWrapper(self.sample_df.copy(), self.id_cols)
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

