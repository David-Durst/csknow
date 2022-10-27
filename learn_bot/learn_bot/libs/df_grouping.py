import pandas as pd
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainTestSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def train_test_split_by_col(df: pd.DataFrame, group_col: str) -> TrainTestSplit:
    # train test split on rounds with rounds weighted by number of entries in each round
    # so 80-20 train test split on actual data with rounds kept coherent
    # split by rounds, weight rounds by number of values in each round
    per_round_df = df.groupby([group_col]).count()

    # sample frac = 1 to shuffle
    random_sum_rounds_df = per_round_df.sample(frac=1).cumsum()

    # top 80% of engagements (summing by ticks per engagement to weight by ticks) are training data, rest are test
    top_80_pct_rounds = random_sum_rounds_df[random_sum_rounds_df['id'] < 0.8 * len(df)].index.to_list()
    all_data_df_split_predicate = df[group_col].isin(top_80_pct_rounds)
    return TrainTestSplit(df[all_data_df_split_predicate], df[~all_data_df_split_predicate])
