import pandas as pd
from pathlib import Path
from learn_bot.libs.df_grouping import train_test_split_by_col
from learn_bot.libs.temporal_column_names import TemporalIOColumnNames
from learn_bot.navigation.dataset import NavDataset
from typing import List

csv_outputs_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs'

non_img_df = pd.read_csv(csv_outputs_path / 'trainNav.csv')

train_test_split = train_test_split_by_col(non_img_df, 'trajectory id')
train_df = train_test_split.train_df
test_df = train_test_split.test_df

base_float_columns: List[str] = ["delta view angle x", "delta view angle y",
                                 "recoil angle x", "recoil angle y",
                                 "delta view angle recoil adjusted x", "delta view angle recoil adjusted y",
                                 "delta position x", "delta position y", "delta position z",
                                 "eye-to-head distance"]

temporal_io_float_column_names = TemporalIOColumnNames(base_float_columns, 0, 0, 0)

train_dataset = NavDataset(train_df, csv_outputs_path / 'trainNavData.tar')
train_dataset.__getitem__(0)
test_dataset = NavDataset(test_df)
