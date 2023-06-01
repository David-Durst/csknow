import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Set, Optional


class HDF5Wrapper:
    hdf5_path: str
    id_cols: List[str]
    id_df: pd.DataFrame
    sample_df: pd.DataFrame
    hdf5_file: h5py.File
    hdf5_data: h5py.Group

    def __init__(self, hdf5_path: str, id_cols: List[str]):
        self.hdf5_path = hdf5_path
        self.id_cols = id_cols
        self.id_df = load_hdf5_to_pd(hdf5_path, cols_to_get=id_cols)
        self.sample_df = load_hdf5_to_pd(hdf5_path, rows_to_get=[i for i in range(min(100, len(self.id_df)))])
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.hdf5_data = self.hdf5_file['data']

    def limit(self, selector_df: pd.Series):
        self.id_df = self.id_df[selector_df]

    def __len__(self):
        return len(self.id_df)


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

