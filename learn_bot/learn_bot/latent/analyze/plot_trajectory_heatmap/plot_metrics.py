from math import ceil
from pathlib import Path
from typing import List, Dict, Set, Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import plot_hist, generate_bins
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmap_plots import get_title_to_speeds, \
    get_title_to_lifetimes, get_title_to_shots_and_kills, kills_column, shots_column

fig_length = 6


def plot_one_metric_histograms(title_to_values: Dict[str, List[float]], metric_title: str,
                               bin_width: Union[int, float], smallest_max: float,
                               x_label: str, plot_file_path: Path):
    num_titles = len(title_to_values.keys())
    fig = plt.figure(figsize=(fig_length, fig_length * num_titles), constrained_layout=True)
    axs = fig.subplots(num_titles, 1, squeeze=False)

    # get the max value for histogram (if bigger than smaller max), min wil be 0 always
    max_observed = smallest_max
    for title, values in title_to_values.items():
        max_observed = max(max_observed, max(values))

    if bin_width < 1.:
        num_bins = int(ceil(max_observed / bin_width))
        # add 1 as need left edge of every bin and right edge of last bin
        bins = [i * bin_width for i in range(num_bins + 1)]
    else:
        bins = generate_bins(0, int(ceil(max_observed)), bin_width)
    ax_index = 0
    for title, values in title_to_values.items():
        plot_hist(axs[ax_index, 0], pd.Series(values), bins)
        axs[ax_index, 0].set_xlim(0., max_observed)
        axs[ax_index, 0].set_ylim(0., 1.)
        axs[ax_index, 0].set_title(title + " " + metric_title)
        axs[ax_index, 0].set_title(title + " " + metric_title)
        axs[ax_index, 0].set_xlabel(x_label)
        ax_index += 1
    plt.savefig(plot_file_path)


def plot_one_metric_bars(title_to_event_a_and_event_b: Dict[str, List[Dict[str, int]]], metric_title: str,
                         x_label: str, y_label: str, x_column_name: str, base_y_column_name: str,
                         plot_file_path: Path):
    # build a dataframe with average y value per x value for each title
    avg_y_per_x_dfs: List[pd.DataFrame] = {}
    title_y_column_names: List[str] = []
    for title, event_a_and_event_b_list in title_to_event_a_and_event_b.items():
        ungrouped_df = pd.DataFrame.from_records(event_a_and_event_b_list)
        title_y_column_names.append(f"{title} {base_y_column_name}")
        ungrouped_df.rename({base_y_column_name: title_y_column_names[-1]}, inplace=True)
        avg_y_per_x_dfs.append(ungrouped_df.groupby(x_column_name, as_index=False).mean())
    avg_y_per_x_df = pd.merge(avg_y_per_x_dfs, on=x_column_name)

    # plot the different columns
    ax = avg_y_per_x_df.plot(x=x_column_name, y=list(title_y_column_names), kind='bar', rot=0)
    ax.set_title(metric_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.get_figure().savefig(plot_file_path)


def plot_metrics(trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    if trajectory_filter_options.compute_speeds:
        # airstrafing can get you above normal weapon max speed
        plot_one_metric_histograms(get_title_to_speeds(), 'Weapon/Scoped Scaled Speed', 0.1, 1., 'Percent Max Speed',
                                   plots_path / ('speeds_' + str(trajectory_filter_options) + '.png'))
    if trajectory_filter_options.compute_lifetimes:
        # small timing mismatch can get 41 seconds on bomb timer
        plot_one_metric_histograms(get_title_to_lifetimes(), 'Lifetimes', 5, 40., 'Lifetime Length (s)',
                                   plots_path / ('lifetimes_' + str(trajectory_filter_options) + '.png'))
    if trajectory_filter_options.compute_shots_per_kill:
        plot_one_metric_bars(get_title_to_shots_and_kills(), 'Shots Per Kill',
                             'Kills in Round', 'Avg Shots in Round', kills_column, shots_column,
                             plots_path / ('shots_per_kill_' + str(trajectory_filter_options) + '.png'))
