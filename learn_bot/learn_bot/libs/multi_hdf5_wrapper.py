from pathlib import Path
from typing import List, Dict, Union, Optional
from dataclasses import dataclass

import pandas as pd

from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.libs.df_grouping import TrainTestSplit, train_test_split_by_col, get_test_col_ids, \
    train_test_split_by_col_ids
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper, PDWrapper
from learn_bot.libs.io_transforms import IOColumnTransformers

HDF5SourceOptions = Union[Path, HDF5Wrapper]


class MultiHDF5Wrapper:
    hdf5_wrappers: List[HDF5Wrapper]
    # subsets of main hdf5_wrappers for train/test splits
    diff_train_test: bool
    force_test_hdf5: bool
    train_hdf5_wrappers: List[HDF5Wrapper]
    test_hdf5_wrappers: List[HDF5Wrapper]
    test_group_ids: Dict[Path, List[int]]
    # these are only valid if diff_train_test is true, otherwise no splits
    train_test_splits: Dict[Path, TrainTestSplit]
    duplicate_last_hdf5_equal_to_rest: bool

    # each source is a path to an hdf5, a directory with hdf5, or an hdf5 wrapper
    def __init__(self, hdf5_sources: List[HDF5SourceOptions], id_cols: List[str], diff_train_test: bool,
                 force_test_hdf5: Optional[HDF5Wrapper] = None, duplicate_last_hdf5_equal_to_rest: bool = False):
        self.hdf5_wrappers = []
        for hdf5_source in hdf5_sources:
            if isinstance(hdf5_source, Path):
                if hdf5_source.is_dir():
                    hdf5_files = hdf5_source.glob('behaviorTreeTeamFeatureStore*.hdf5')
                    # only need sample df from first file, rest can just be empty
                    empty_like_first_sample_df: Optional[pd.DataFrame] = None
                    for hdf5_file in hdf5_files:
                        self.hdf5_wrappers.append(HDF5Wrapper(hdf5_file, id_cols, sample_df=empty_like_first_sample_df))
                        if empty_like_first_sample_df is None:
                            empty_like_first_sample_df = pd.DataFrame().reindex_like(self.hdf5_wrappers[0].sample_df)
                        if len(self.hdf5_wrappers) > 0:
                            break
                elif hdf5_source.is_file() and hdf5_source.name.endswith('.hdf5'):
                    self.hdf5_wrappers.append(HDF5Wrapper(hdf5_source, id_cols))
            elif isinstance(hdf5_source, HDF5Wrapper):
                self.hdf5_wrappers.append(hdf5_source)
            else:
                raise Exception("MultiHDF5Wrapper initialized with source that isn't path or PDWrapper")
        self.train_hdf5_wrappers = []
        self.test_hdf5_wrappers = []
        self.test_group_ids = {}
        self.diff_train_test = diff_train_test
        self.force_test_hdf5 = force_test_hdf5
        self.duplicate_last_hdf5_equal_to_rest = duplicate_last_hdf5_equal_to_rest

    def train_test_split_by_col(self, force_test_hdf5: Optional[HDF5Wrapper]):
        self.train_test_splits = {}
        for hdf5_wrapper in self.hdf5_wrappers:
            if self.diff_train_test:
                train_test_split = train_test_split_by_col(hdf5_wrapper.id_df, round_id_column)
                self.record_train_test_split(hdf5_wrapper, train_test_split)
            else:
                self.train_hdf5_wrappers.append(hdf5_wrapper)
        if not self.diff_train_test:
            if force_test_hdf5:
                self.test_hdf5_wrappers.append(force_test_hdf5)
            else:
                self.test_hdf5_wrappers = self.train_hdf5_wrappers

    def train_test_split_by_col_id(self, train_test_splits):
        self.train_test_splits = train_test_splits
        for hdf5_wrapper in self.hdf5_wrappers:
            if self.diff_train_test:
                train_test_split = train_test_split_by_col_ids(hdf5_wrapper.id_df, round_id_column,
                                                               train_test_splits[hdf5_wrapper.hdf5_path])
                self.record_train_test_split(hdf5_wrapper, train_test_split)

    def record_train_test_split(self, hdf5_wrapper: HDF5Wrapper, train_test_split: TrainTestSplit):
        self.train_test_splits[hdf5_wrapper.hdf5_path] = train_test_split
        train_hdf5_wrapper = hdf5_wrapper.clone()
        train_hdf5_wrapper.limit(train_test_split.train_predicate)
        self.train_hdf5_wrappers.append(train_hdf5_wrapper)
        test_hdf5_wrapper = hdf5_wrapper.clone()
        test_hdf5_wrapper.limit(~train_test_split.train_predicate)
        self.test_hdf5_wrappers.append(test_hdf5_wrapper)
        self.test_group_ids[hdf5_wrapper.hdf5_path] = get_test_col_ids(test_hdf5_wrapper.id_df, round_id_column)

    def create_np_arrays(self, cts: IOColumnTransformers, load_output_data: bool = True):
        for hdf5_wrapper in self.hdf5_wrappers:
            hdf5_wrapper.create_np_array(cts, load_output_data)
        # if forcing test hdf5, load it's unique np arrays
        if self.force_test_hdf5:
            for hdf5_wrapper in self.test_hdf5_wrappers:
                hdf5_wrapper.create_np_array(cts, load_output_data)


