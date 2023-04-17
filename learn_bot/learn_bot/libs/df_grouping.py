import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainTestSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_group_ids: List[int]


def train_test_split_by_col(df: pd.DataFrame, group_col: str) -> TrainTestSplit:
    # train test split on group col with groups weighted by number of entries in each round
    # so 80-20 train test split on actual data with groups kept coherent
    # split by groups, weight groups by number of values in each groups
    per_group_df = df.groupby([group_col]).count()

    # sample frac = 1 to shuffle
    random_sum_groups_df = per_group_df.sample(frac=1).cumsum()

    # top 80% of engagements (summing by ticks per engagement to weight by ticks) are training data, rest are test
    top_80_pct_groups = random_sum_groups_df[random_sum_groups_df['id'] < 0.8 * len(df)].index.to_list()
    all_data_df_split_predicate = df[group_col].isin(top_80_pct_groups)
    return TrainTestSplit(df[all_data_df_split_predicate], df[~all_data_df_split_predicate], top_80_pct_groups)


def train_test_split_by_col_ids(df: pd.DataFrame, group_col: str, col_ids: List[int]) -> TrainTestSplit:
    all_data_df_split_predicate = df[group_col].isin(col_ids)
    return TrainTestSplit(df[all_data_df_split_predicate], df[~all_data_df_split_predicate], col_ids)


def get_test_col_ids(train_test_split: TrainTestSplit, group_col: str) -> List[int]:
    return train_test_split.train_df.groupby([group_col]).index.to_list()


def make_index_column(df: pd.DataFrame):
    df.drop("index", axis=1, errors="ignore", inplace=True)
    # creates a new index based on columns (index can be invalid if produced by train/test)
    df.reset_index(inplace=True, drop=True)
    # makes new index into column
    df.reset_index(inplace=True, drop=False)


def get_row_as_dict_iloc(df: pd.DataFrame, row_index: int) -> Dict:
    return df.iloc[[row_index], :].to_dict(orient='records')[0]


def get_row_as_dict_loc(df: pd.DataFrame, row_index: int) -> Dict:
    return df.loc[[row_index], :].to_dict(orient='records')[0]
