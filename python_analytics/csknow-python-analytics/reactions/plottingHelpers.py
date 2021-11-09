import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import math
import numpy as np

def makePlotterFunction(bin_width, pct):
    def plotPct(df, col_name, ax):
        col_vals = df[col_name].dropna()
        num_bins = math.ceil((col_vals.max() - col_vals.min()) / bin_width) + 1
        df.hist(col_name, bins=num_bins, ax=ax, weights=np.ones(len(col_vals)) / len(col_vals))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        return pd.cut(col_vals, num_bins).value_counts(normalize=True).sort_index()

    def plotNum(df, col_name, ax):
        col_vals = df[col_name].dropna()
        num_bins = math.ceil((col_vals.max() - col_vals.min()) / bin_width) + 1
        df.hist(col_name, bins=num_bins, ax=ax)
        return pd.cut(col_vals, num_bins).value_counts().sort_index()

    if pct:
        return plotPct
    else:
        return plotNum


def makeHistograms(dfs, col_name, plotting_function, plot_titles, name, x_label, plot_folder):
    num_rows = len(dfs)
    num_cols = len(dfs[0])
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*8, num_rows*8))

    distributions = []
    for r in range(num_rows):
        distributions.append([])
        for c in range(num_cols):
            if len(dfs[r][c]) == 0:
                continue
            distributions[r].append(plotting_function(dfs[r][c], col_name, ax[r][c]))

    def get_num_points_coordinate(ax, x_pct = 0.6, y_pct = 0.8):
        (y_min, y_max) = ax.get_ylim()
        (x_min, x_max) = ax.get_xlim()
        return (x_min + (x_max-x_min) * x_pct, y_min + (y_max-y_min)*y_pct)

    for r in range(num_rows):
        xmin = 10000000
        xmax = -1 * xmin
        ymin = 10000000
        ymax = -1 * ymin
        for c in range(num_cols):
            if len(dfs[r][c]) == 0:
                continue
            cur_xlim = ax[r][c].get_xlim()
            xmin = min(xmin, cur_xlim[0])
            xmax = max(xmax, cur_xlim[1])
            cur_ylim = ax[r][c].get_ylim()
            ymin = min(ymin, cur_ylim[0])
            ymax = max(ymax, cur_ylim[1])

        for c in range(num_cols):
            ax[r][c].set_xlim(xmin, xmax)
            ax[r][c].set_ylim(0, ymax)
            ax[r][c].set_xlabel(x_label, fontsize=14)
            ax[r][c].set_ylabel('Frequency', fontsize=14)
            ax[r][c].set_title(plot_titles[r][c], fontsize=18)
            ax[r][c].annotate('total points: ' + str(len(dfs[r][c].dropna())),
                              get_num_points_coordinate(ax[r][c]), fontsize="14")

    plt.suptitle(name, fontsize=30)
    plt.tight_layout()
    fig.savefig(plot_folder + 'hist_' + name.lower().replace(' ', '_') + '.png')

    plt.clf()
