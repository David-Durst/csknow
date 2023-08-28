from math import ceil
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.humanness_metrics.hdf5_loader import HumannessMetrics, HumannessDataOptions
from learn_bot.latent.analyze.process_trajectory_comparison import plot_hist, generate_bins, set_pd_print_options
from learn_bot.latent.analyze.test_traces.column_names import rollout_aggressive_humanness_hdf5_data_path, \
    rollout_passive_humanness_hdf5_data_path

fig_length = 6
num_figs = 28
num_player_types = 2
aggressive_learned_bot_name = "Aggressive Learned Bot "
passive_learned_bot_name = "Passive Learned Bot "


def plot_metric(axs, metric_index: int, aggressive_learned_bot_metric: np.ndarray,
                passive_learned_bot_metric: np.ndarray, metric_name: str, pct_bins: bool = False):
    max_value = int(ceil(max(aggressive_learned_bot_metric.max(), passive_learned_bot_metric.max())))
    axs[metric_index, 0].set_title(aggressive_learned_bot_name + metric_name)
    axs[metric_index, 1].set_title(passive_learned_bot_name + metric_name)
    bins: List
    if pct_bins:
        bins = [i * 0.1 for i in range(11)]
    else:
        bins = generate_bins(0, max_value, max_value // 20)
    plot_hist(axs[metric_index, 0], pd.Series(aggressive_learned_bot_metric), bins)
    plot_hist(axs[metric_index, 1], pd.Series(passive_learned_bot_metric), bins)
    axs[metric_index, 0].set_ylim(0., 1.)
    axs[metric_index, 1].set_ylim(0., 1.)


def plot_humanness_metrics():
    aggressive_humanness_metrics = HumannessMetrics(HumannessDataOptions.CUSTOM, False,
                                                    rollout_aggressive_humanness_hdf5_data_path)
    rollout_humanness_metrics = HumannessMetrics(HumannessDataOptions.CUSTOM, False,
                                                 rollout_passive_humanness_hdf5_data_path)

    set_pd_print_options()


    fig = plt.figure(figsize=(fig_length*num_player_types, fig_length*num_figs), constrained_layout=True)
    fig.suptitle("Aggressive Learned Bot vs Passive Learned Bot Metrics")
    axs = fig.subplots(num_figs, num_player_types, squeeze=False)
