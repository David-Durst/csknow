import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from pandas.core.dtypes.common import is_numeric_dtype


def fix_bool_cols(df: pd.DataFrame):
    for col in df.columns:
        if str(df[col].dtype) == 'bool':
            df[col] = df[col].astype(int)


def load_hdf5_to_pd(hdf5_path: Path):
    # get data as numpy arrays and column names
    #np_arrs: List[np.ndarray] = []
    #col_names: List[List[str]] = []
    partial_dfs: List[pd.DataFrame] = []
    with h5py.File(hdf5_path) as hdf5_file:
        hdf5_data = hdf5_file['data']
        flat_col_names: list[str] = []
        for k in hdf5_data.keys():
            if hdf5_data[k].attrs:
                np_arr: np.ndarray = hdf5_data[k][...]
                col_names = hdf5_data[k].attrs['column names'].split(',')
                partial_dfs.append(pd.DataFrame(np_arr, columns=col_names))
            else:
                flat_col_names += k

        if len(flat_col_names) > 0:
            np_arr: np.ndarray = hdf5_data[flat_col_names][...]
            partial_dfs.append(pd.DataFrame(np_arr, columns=flat_col_names))

    if len(partial_dfs) == 1:
        fix_bool_cols(partial_dfs[0])
        return partial_dfs[0]
    else:
        result_df = pd.concat(partial_dfs, axis=1)
        fix_bool_cols(result_df)
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

