import math
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

INCH_PER_FIG = 4

def filter_df(df: pd.DataFrame, col_name, low_pct_to_remove=0.01, high_pct_to_remove=0.01) -> pd.DataFrame:
    q_low = df[col_name].quantile(low_pct_to_remove)
    q_hi = df[col_name].quantile(1. - high_pct_to_remove)
    return df[(df[col_name] <= q_hi) & (df[col_name] >= q_low)]


def filter_df_2d(df: pd.DataFrame, col1_name, col2_name,
                 low_pct_to_remove=0.01, high_pct_to_remove=0.01) -> pd.DataFrame:
    q1_low = df[col1_name].quantile(low_pct_to_remove)
    q1_hi = df[col1_name].quantile(1. - high_pct_to_remove)
    q2_low = df[col2_name].quantile(low_pct_to_remove)
    q2_hi = df[col2_name].quantile(1. - high_pct_to_remove)
    return df[(df[col1_name] <= q1_hi) & (df[col1_name] >= q1_low) &
              (df[col2_name] <= q2_hi) & (df[col2_name] >= q2_low)]


def plot_untransformed_and_transformed(plot_path: Path, title: str, df, float_cols, cat_cols,
                                       transformed_df = None):
    # plot untransformed and transformed outputs
    num_float_cols = len(float_cols)
    num_float_rows = 1
    if num_float_cols > 16:
        num_float_rows = math.ceil(math.sqrt(num_float_cols))
        num_float_cols = num_float_rows

    # untransformed
    fig = plt.figure(figsize=(INCH_PER_FIG * 2 * num_float_cols, num_float_rows * INCH_PER_FIG), constrained_layout=True)
    axs = fig.subplots(num_float_rows, num_float_cols, squeeze=False)
    fig.suptitle(title + ' float untransformed')
    for i in range(num_float_rows):
        axs[i][0].set_ylabel('num points')

    with tqdm(total=len(float_cols), disable=False) as pbar:
        for i in range(len(float_cols)):
            df.hist(float_cols[i], ax=axs[i // num_float_cols][i % num_float_cols], bins=100)
            #axs[0][i].set_xlabel(float_column_x_axes[i % len(float_column_x_axes)])
            pbar.update(1)
    plt.savefig(plot_path / (title + ' untransformed.png'))

    # transformed
    if transformed_df is not None:
        fig = plt.figure(figsize=(INCH_PER_FIG * 2 * num_float_cols, num_float_rows * INCH_PER_FIG),
                         constrained_layout=True)
        axs = fig.subplots(num_float_rows, num_float_cols, squeeze=False)
        fig.suptitle(title + ' float transformed')
        for i in range(num_float_rows):
            axs[i][0].set_ylabel('num points')
        for i in range(len(float_cols)):
            transformed_df_filtered = filter_df(transformed_df, float_cols[i])
            transformed_df_filtered.hist(float_cols[i], ax=axs[i // num_float_cols][i % num_float_cols], bins=100)
            #axs[0][i].set_xlabel(float_column_x_axes[i % len(float_column_x_axes)] + ' standardized')
        plt.savefig(plot_path / (title + ' transformed.png'))

    # categorical
    fig = plt.figure(figsize=(INCH_PER_FIG * 2 * 10, 2.5 * INCH_PER_FIG), constrained_layout=True)
    if cat_cols:
        axs = fig.subplots(1, len(cat_cols), squeeze=False)
        fig.suptitle(title + ' categorical')
        axs[0][0].set_ylabel('num points')
        for i in range(len(cat_cols)):
            #axs[0][i].set_xlabel(cat_column_x_axes[i % len(cat_column_x_axes)])
            df.loc[:, cat_cols[i]].value_counts().plot.bar(ax=axs[0][i], title=cat_cols[i])
        plt.savefig(plot_path / (title + ' cat.png'))
