from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

INCH_PER_FIG = 4
plot_path = Path(__file__).parent / 'distributions'


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


def plot_untransformed_and_transformed(title: str, df, float_cols, cat_cols,
                                       transformed_df = None):
    # plot untransformed and transformed outputs
    fig = plt.figure(figsize=(INCH_PER_FIG * len(float_cols), 3.5 * INCH_PER_FIG), constrained_layout=True)
    fig.suptitle(title)
    num_rows = 1
    if transformed_df is not None:
        num_rows += 1
    if cat_cols:
        num_rows += 1
    subfigs = fig.subfigures(nrows=num_rows, ncols=1, squeeze=False)

    # untransformed
    axs = subfigs[0][0].subplots(1, len(float_cols), squeeze=False)
    subfigs[0][0].suptitle('float untransformed')
    axs[0][0].set_ylabel('num points')
    for i in range(len(float_cols)):
        df_filtered = filter_df(df, float_cols[i])
        df_filtered.hist(float_cols[i], ax=axs[0][i], bins=100)
        #axs[0][i].set_xlabel(float_column_x_axes[i % len(float_column_x_axes)])

    # transformed
    if transformed_df is not None:
        axs = subfigs[1][0].subplots(1, len(float_cols), squeeze=False)
        subfigs[1][0].suptitle('float transformed')
        axs[0][0].set_ylabel('num points')
        for i in range(len(float_cols)):
            transformed_df_filtered = filter_df(transformed_df, float_cols[i])
            transformed_df_filtered.hist(float_cols[i], ax=axs[0][i], bins=100)
            #axs[0][i].set_xlabel(float_column_x_axes[i % len(float_column_x_axes)] + ' standardized')

    # categorical
    if cat_cols:
        axs = subfigs[num_rows-1][0].subplots(1, len(cat_cols), squeeze=False)
        subfigs[num_rows-1][0].suptitle('categorical')
        axs[0][0].set_ylabel('num points')
        for i in range(len(cat_cols)):
            #axs[0][i].set_xlabel(cat_column_x_axes[i % len(cat_column_x_axes)])
            df.loc[:, cat_cols[i]].value_counts().plot.bar(ax=axs[0][i], title=cat_cols[i])
    plt.savefig(plot_path / (title + '.png'))
