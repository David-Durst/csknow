from math import ceil
from pathlib import Path
from typing import List, Dict, Set, Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import plot_hist, generate_bins
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import get_title_to_speeds, \
    get_title_to_lifetimes, get_title_to_shots_per_kill

fig_length = 6


def compute_one_metric_histograms(title_to_values: Dict[str, List[float]], metric_title: str,
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


def compute_metrics(trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    if trajectory_filter_options.compute_speeds:
        # airstrafing can get you above normal weapon max speed
        compute_one_metric_histograms(get_title_to_speeds(), 'Weapon/Scoped Scaled Speed', 0.1, 1., 'Percent Max Speed',
                                      plots_path / ('speeds_' + str(trajectory_filter_options) + '.png'))
    if trajectory_filter_options.compute_lifetimes:
        # small timing mismatch can get 41 seconds on bomb timer
        compute_one_metric_histograms(get_title_to_lifetimes(), 'Lifetimes', 5, 40., 'Lifetime Length (s)',
                                      plots_path / ('lifetimes_' + str(trajectory_filter_options) + '.png'))
    if trajectory_filter_options.compute_shots_per_kill:
        compute_one_metric_histograms(get_title_to_shots_per_kill(), 'Shots Per Kill', 1, 30., 'Shots Per Kill',
                                      plots_path / ('shots_per_kill_' + str(trajectory_filter_options) + '.png'))
