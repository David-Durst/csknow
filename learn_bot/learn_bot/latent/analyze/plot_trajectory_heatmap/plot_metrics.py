from math import ceil
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import plot_hist, generate_bins
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.plot_trajectory_heatmap.plot_one_trajectory_np import get_title_to_speeds, get_title_to_lifetimes

fig_length = 6


def plot_one_metric(title_to_values: Dict[str, List[float]], title: str, smallest_max: float,
                    plot_file_path: Path):
    num_titles = len(title_to_values.keys())
    fig = plt.figure(figsize=(fig_length * num_titles, fig_length), constrained_layout=True)
    fig.suptitle(title)
    axs = fig.subplots(num_titles, 1, squeeze=False)

    # get the max value for histogram (if bigger than smaller max), min wil be 0 always
    max_observed = smallest_max
    for title, values in title_to_values.items():
        max_observed = max(max_observed, max(values))

    bins = generate_bins(0, int(ceil(max_observed)), 50)
    ax_index = 0
    for title, values in title_to_values.items():
        plot_hist(axs[ax_index, 0], pd.Series(values), bins)
        axs[ax_index, 0].set_xlim(0., max_observed)
        axs[ax_index, 0].set_ylim(0., 1.)
        ax_index += 1
    plt.savefig(plot_file_path)


def plot_metrics(trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    if trajectory_filter_options.compute_speeds:
        plot_one_metric(get_title_to_speeds(), 'Speed', 250.,
                        plots_path / ('speeds_' + str(trajectory_filter_options) + '.png'))
    if trajectory_filter_options.compute_lifetimes:
        plot_one_metric(get_title_to_lifetimes(), 'Lifetimes', 40.,
                        plots_path / ('lifetimes_' + str(trajectory_filter_options) + '.png'))
