import pandas as pd
from pathlib import Path
from learn_bot.libs.df_grouping import train_test_split_by_col
from learn_bot.libs.temporal_column_names import TemporalIOColumnNames
from learn_bot.navigation.dataset import NavDataset
from learn_bot.navigation.io_transforms import PRIOR_TICKS, CUR_TICK, FUTURE_TICKS, ColumnTypes, \
    IOColumnAndImageTransformers
from typing import List
import time


start_time = time.perf_counter()

csv_outputs_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs'

non_img_df = pd.read_csv(csv_outputs_path / 'trainNav.csv')

train_test_split = train_test_split_by_col(non_img_df, 'trajectory id')
train_df = train_test_split.train_df
test_df = train_test_split.test_df

# create all training data column names
# little clunky as the column generator assumes that t>=0 is label and all t=0 are to be vis'd
base_prior_img_columns: List[str] = ["player pos", "player vis", "player vis from",
                                     "distance map", "friendly pos", "friendly vis", "friendly vis from",
                                     "vis enemies", "vis from enemies", "c4 pos"]
prior_temporal_img_columns = TemporalIOColumnNames(base_prior_img_columns, PRIOR_TICKS, 0, 0)

base_future_img_columns: List[str] = ["goal pos"]
future_temporal_img_columns = TemporalIOColumnNames(base_prior_img_columns, PRIOR_TICKS, 0, 0)

# looking into the future for goal region (higher level model will give us this)
temporal_img_column_names = TemporalIOColumnNames()
temporal_img_column_names.input_columns = prior_temporal_img_columns.input_columns + \
                                          future_temporal_img_columns.output_columns
temporal_img_column_names.vis_columns = temporal_img_column_names.input_columns
temporal_img_column_names.output_columns = []

base_prior_float_columns: List[str] = ["player view dir x", "player view dir y", "health", "armor"]
prior_temporal_float_columns = TemporalIOColumnNames(base_prior_float_columns, PRIOR_TICKS, 0, 0)

base_cur_cat_columns: List[str] = ["movement result x", "movement result y"]
cur_temporal_cat_columns = TemporalIOColumnNames(base_cur_cat_columns, 0, CUR_TICK, 0)

# transform input and output
input_column_types = ColumnTypes(prior_temporal_float_columns.input_columns, [], [])

output_column_types = ColumnTypes([], cur_temporal_cat_columns.output_columns, [3, 3])

# weird dance as column transformers need data set to compute mean/std dev
# but data set needs column transformers during train/inference time
# so make data set without transformers, then use dataset during transformer creation, then pass
# transformers to dataset
nav_dataset = NavDataset(non_img_df, csv_outputs_path / 'trainNavData.tar', temporal_img_column_names.vis_columns)
column_transformers = IOColumnAndImageTransformers(input_column_types, output_column_types, non_img_df, )


# make all the columns input/vis, since looking into future




base_float_columns: List[str] = ["delta view angle x", "delta view angle y",
                                 "recoil angle x", "recoil angle y",
                                 "delta view angle recoil adjusted x", "delta view angle recoil adjusted y",
                                 "delta position x", "delta position y", "delta position z",
                                 "eye-to-head distance"]

temporal_io_float_column_names = TemporalIOColumnNames(base_float_columns, 0, 0, 0)

train_dataset = NavDataset(train_df, csv_outputs_path / 'trainNavData.tar')
train_dataset.__getitem__(0)
test_dataset = NavDataset(test_df)
