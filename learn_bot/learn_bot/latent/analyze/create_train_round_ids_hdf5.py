import pickle
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

from learn_bot.latent.train_paths import train_test_split_file_name
from learn_bot.libs.df_grouping import TrainTestSplit
from learn_bot.libs.multi_hdf5_wrapper import train_test_split_folder_path, make_train_test_splits_relative

train_round_ids_file_name = 'train_round_ids.hdf5'

with open(train_test_split_folder_path / train_test_split_file_name, "rb") as pickle_file:
    train_test_splits: Dict[Path, TrainTestSplit] = pickle.load(pickle_file)


with h5py.File(train_test_split_folder_path / train_round_ids_file_name, 'w') as hdf5_file:
    for path, train_test_split in train_test_splits.items():
        group_name = path.parent.name
        data_set_name = path.name
        group = hdf5_file.require_group(str(group_name))
        group.create_dataset(data_set_name, data=np.asarray(train_test_split.train_group_ids))
