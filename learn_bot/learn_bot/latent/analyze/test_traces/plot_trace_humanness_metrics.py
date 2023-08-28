from math import ceil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.humanness_metrics.hdf5_loader import HumannessMetrics, HumannessDataOptions
from learn_bot.latent.analyze.process_trajectory_comparison import plot_hist, generate_bins, set_pd_print_options
from learn_bot.latent.analyze.test_traces.column_names import rollout_aggressive_humanness_hdf5_data_path, \
    rollout_passive_humanness_hdf5_data_path, trace_index_name, trace_one_non_replay_team_name, \
    trace_one_non_replay_bot_name, rollout_aggressive_trace_hdf5_data_path, rollout_passive_trace_hdf5_data_path, \
    trace_demo_file_name
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

fig_length = 6
num_figs = 28
num_player_types = 2
aggressive_learned_bot_name = "Aggressive Learned Bot "
passive_learned_bot_name = "Passive Learned Bot "


def get_trace_png_name(trace_index: int, trace_demo_file: str, one_non_replay_bot: bool) -> str:
    return str(trace_index) + "_" + trace_demo_file + "_" + str(one_non_replay_bot) + ".png"


def get_round_data_df(trace_hdf5_data_path: Path):
    trace_df = load_hdf5_to_pd(trace_hdf5_data_path)
    subset_trace_df = trace_df.loc[:, [round_id_column, trace_demo_file_name, trace_index_name,
                                       trace_one_non_replay_team_name, trace_one_non_replay_bot_name]]
    return subset_trace_df.groubpy(round_id_column, index=False).agg({trace_demo_file_name: 'first',
                                                                      trace_index_name: 'first',
                                                                      trace_one_non_replay_team_name: 'first',
                                                                      trace_one_non_replay_bot_name: 'first'})


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
    set_pd_print_options()

    aggressive_humanness_metrics = HumannessMetrics(HumannessDataOptions.CUSTOM, False,
                                                    rollout_aggressive_humanness_hdf5_data_path)
    rollout_humanness_metrics = HumannessMetrics(HumannessDataOptions.CUSTOM, False,
                                                 rollout_passive_humanness_hdf5_data_path)

    aggressive_round_data_df = get_round_data_df(rollout_aggressive_trace_hdf5_data_path)
    passive_round_data_df = get_round_data_df(rollout_passive_trace_hdf5_data_path)

    for trace_index in list(aggressive_round_data_df[trace_index_name].unique()):
        # TODO: get demo file name
        trace_demo_file = cur_trace_extra_df.loc[cur_round_id, trace_demo_file_name].decode('utf-8')[:-1]

        fig = plt.figure(figsize=(fig_length * num_player_types, fig_length * num_figs), constrained_layout=True)
        fig.suptitle("Aggressive Learned Bot vs Passive Learned Bot Humannesss Metrics " + )
        axs = fig.subplots(num_figs, num_player_types, squeeze=False)

        aggressive_trace_round_ids = list(
            aggressive_round_data_df[aggressive_round_data_df[trace_index_name] == trace_index].loc[:, round_id_column]
        )
        passive_trace_round_ids = list(
            passive_round_data_df[passive_round_data_df[trace_index_name] == trace_index].loc[:, round_id_column]
        )

        # TODO: get metrics for the round ids
        aggressive_trace_humanness_metrics = aggressive_humanness_metrics





