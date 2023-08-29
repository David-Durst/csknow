import os
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.humanness_metrics.column_names import distance_to_nearest_teammate_name, \
    distance_to_nearest_teammate_when_firing_name, distance_to_nearest_enemy_when_shot_name, \
    delta_distance_to_nearest_teammate_name, delta_distance_to_nearest_teammate_when_firing_name, \
    delta_distance_to_c4_when_shot_name, \
    distance_to_c4_name, distance_to_c4_when_firing_name, distance_to_c4_when_shot_name, \
    delta_distance_to_c4_name, delta_distance_to_c4_when_firing_name, delta_distance_to_c4_when_shot_name, \
    distance_to_cover_when_enemy_visible_fov_name, distance_to_cover_when_firing_name, distance_to_cover_when_shot_name, \
    time_from_firing_to_teammate_seeing_enemy_fov_name, time_from_shot_to_teammate_seeing_enemy_fov_name, \
    distance_to_c4_when_enemy_visible_fov_name, delta_distance_to_c4_when_enemy_visible_fov_name

from learn_bot.latent.analyze.humanness_metrics.hdf5_loader import HumannessMetrics, HumannessDataOptions
from learn_bot.latent.analyze.process_trajectory_comparison import plot_hist, generate_bins, set_pd_print_options
from learn_bot.latent.analyze.test_traces.column_names import rollout_aggressive_humanness_hdf5_data_path, \
    rollout_passive_humanness_hdf5_data_path, trace_index_name, trace_one_non_replay_team_name, \
    trace_one_non_replay_bot_name, rollout_aggressive_trace_hdf5_data_path, rollout_passive_trace_hdf5_data_path, \
    trace_demo_file_name, num_traces_name, trace_is_bot_player_names, trace_humanness_path
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

RoundBotPlayerIds = Dict[int, List[int]]
TraceBotPlayerIds = Dict[int, RoundBotPlayerIds]

metric_cols = ['distance_to_nearest_teammate', 'distance_to_nearest_teammate_when_firing', 'distance_to_nearest_teammate_when_shot',
               'delta_distance_to_nearest_teammate', 'delta_distance_to_nearest_teammate_when_firing', 'delta_distance_to_nearest_teammate_when_shot',
               'distance_to_c4', 'distance_to_c4_when_enemy_visible_fov', 'distance_to_c4_when_firing', 'distance_to_c4_when_shot',
               'delta_distance_to_c4', 'delta_distance_to_c4_when_enemy_visible_fov', 'delta_distance_to_c4_when_firing', 'delta_distance_to_c4_when_shot',
               'distance_to_cover_when_enemy_visible_fov', 'distance_to_cover_when_firing', 'distance_to_cover_when_shot',
               'time_from_firing_to_teammate_seeing_enemy_fov', 'time_from_shot_to_teammate_seeing_enemy_fov']

metric_names = [distance_to_nearest_teammate_name, distance_to_nearest_teammate_when_firing_name, distance_to_nearest_enemy_when_shot_name,
                delta_distance_to_nearest_teammate_name, delta_distance_to_nearest_teammate_when_firing_name, delta_distance_to_c4_when_shot_name,
                distance_to_c4_name, distance_to_c4_when_enemy_visible_fov_name, distance_to_c4_when_firing_name, distance_to_c4_when_shot_name,
                delta_distance_to_c4_name, delta_distance_to_c4_when_enemy_visible_fov_name, delta_distance_to_c4_when_firing_name, delta_distance_to_c4_when_shot_name,
                distance_to_cover_when_enemy_visible_fov_name, distance_to_cover_when_firing_name, distance_to_cover_when_shot_name,
                time_from_firing_to_teammate_seeing_enemy_fov_name, time_from_shot_to_teammate_seeing_enemy_fov_name]

round_id_cols = ['round_id_per_nearest_teammate', 'round_id_per_nearest_teammate_firing', 'round_id_per_nearest_teammate_shot',
                 'round_id_per_nearest_teammate', 'round_id_per_nearest_teammate_firing', 'round_id_per_nearest_teammate_shot',
                 'round_id_per_pat', 'round_id_per_enemy_visible_fov_pat', 'round_id_per_firing_pat', 'round_id_per_shot_pat',
                 'round_id_per_pat', 'round_id_per_enemy_visible_fov_pat', 'round_id_per_firing_pat', 'round_id_per_shot_pat',
                 'round_id_per_enemy_visible_fov_pat', 'round_id_per_firing_pat', 'round_id_per_shot_pat',
                 'round_id_per_firing_to_teammate_seeing_enemy', 'round_id_per_shot_to_teammate_seeing_enemy']

player_id_cols = ['player_id_per_nearest_teammate', 'player_id_per_nearest_teammate_firing', 'player_id_per_nearest_teammate_shot',
                  'player_id_per_nearest_teammate', 'player_id_per_nearest_teammate_firing', 'player_id_per_nearest_teammate_shot',
                  'player_id_per_pat', 'player_id_per_enemy_visible_fov_pat', 'player_id_per_firing_pat', 'player_id_per_shot_pat',
                  'player_id_per_pat', 'player_id_per_enemy_visible_fov_pat', 'player_id_per_firing_pat', 'player_id_per_shot_pat',
                  'player_id_per_enemy_visible_fov_pat', 'player_id_per_firing_pat', 'player_id_per_shot_pat',
                  'player_id_per_firing_to_teammate_seeing_enemy', 'player_id_per_shot_to_teammate_seeing_enemy']

fig_length = 8
num_figs = len(metric_cols)
num_player_types = 4
one_bot_aggressive_learned_bot_name = "One Bot Aggressive "
one_team_aggressive_learned_bot_name = "One Team Aggressive "
one_bot_passive_learned_bot_name = "One Bot Passive "
one_team_passive_learned_bot_name = "One Team Passive "


def plot_metric(axs, metric_index: int, one_bot_aggressive_learned_bot_metric: np.ndarray,
                one_team_aggressive_learned_bot_metric: np.ndarray,
                one_bot_passive_learned_bot_metric: np.ndarray,
                one_team_passive_learned_bot_metric: np.ndarray,
                metric_name: str, pct_bins: bool = False):
    values_for_max = []
    if len(one_bot_aggressive_learned_bot_metric) > 0:
        values_for_max.append(one_bot_aggressive_learned_bot_metric.max())
    if len(one_team_aggressive_learned_bot_metric) > 0:
        values_for_max.append(one_team_aggressive_learned_bot_metric.max())
    if len(one_bot_passive_learned_bot_metric) > 0:
        values_for_max.append(one_bot_passive_learned_bot_metric.max())
    if len(one_team_passive_learned_bot_metric) > 0:
        values_for_max.append(one_team_passive_learned_bot_metric.max())
    if len(values_for_max) == 0:
        return
    max_value = int(ceil(max(values_for_max)))
    if max_value == 0:
        return
    axs[metric_index, 0].set_title(one_bot_aggressive_learned_bot_name + metric_name)
    axs[metric_index, 1].set_title(one_team_aggressive_learned_bot_name + metric_name)
    axs[metric_index, 2].set_title(one_bot_passive_learned_bot_name + metric_name)
    axs[metric_index, 3].set_title(one_team_passive_learned_bot_name + metric_name)
    bins: List
    if pct_bins:
        bins = [i * 0.1 for i in range(11)]
    else:
        bins = generate_bins(0, max_value, max(max_value // 20, 1))
    if len(one_bot_aggressive_learned_bot_metric) > 0:
        plot_hist(axs[metric_index, 0], pd.Series(one_bot_aggressive_learned_bot_metric), bins)
    if len(one_team_aggressive_learned_bot_metric) > 0:
        plot_hist(axs[metric_index, 1], pd.Series(one_team_aggressive_learned_bot_metric), bins)
    if len(one_bot_passive_learned_bot_metric) > 0:
        plot_hist(axs[metric_index, 2], pd.Series(one_bot_passive_learned_bot_metric), bins)
    if len(one_team_passive_learned_bot_metric) > 0:
        plot_hist(axs[metric_index, 3], pd.Series(one_team_passive_learned_bot_metric), bins)
    axs[metric_index, 0].set_ylim(0., 1.)
    axs[metric_index, 1].set_ylim(0., 1.)
    axs[metric_index, 2].set_ylim(0., 1.)
    axs[metric_index, 3].set_ylim(0., 1.)


def get_metrics(humanness_metrics: HumannessMetrics, round_ids: List[int], bot_player_ids: RoundBotPlayerIds) \
        -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for i in range(len(metric_cols)):
        metric_col = humanness_metrics.__getattribute__(metric_cols[i])
        round_ids_col = humanness_metrics.__getattribute__(round_id_cols[i])
        player_ids_col = humanness_metrics.__getattribute__(player_id_cols[i])
        # start with false, because building up or's
        conditions = round_ids_col == -1
        for round_id in round_ids:
            conditions = conditions | ((round_ids_col == round_id) & (np.isin(player_ids_col, bot_player_ids[round_id])))
        result[metric_names[i]] = metric_col[conditions]
    return result


def plot_humanness_metrics(aggressive_trace_bot_player_ids: TraceBotPlayerIds,
                           passive_trace_bot_player_ids: TraceBotPlayerIds):
    set_pd_print_options()

    aggressive_humanness_metrics = HumannessMetrics(HumannessDataOptions.CUSTOM, False,
                                                    rollout_aggressive_humanness_hdf5_data_path)
    aggressive_trace_extra_df = load_hdf5_to_pd(rollout_aggressive_trace_hdf5_data_path, root_key='extra',
                                                cols_to_get=[trace_demo_file_name, trace_index_name, num_traces_name,
                                                             trace_one_non_replay_team_name,
                                                             trace_one_non_replay_bot_name] + trace_is_bot_player_names)
    passive_humanness_metrics = HumannessMetrics(HumannessDataOptions.CUSTOM, False,
                                                 rollout_passive_humanness_hdf5_data_path)
    passive_trace_extra_df = load_hdf5_to_pd(rollout_passive_trace_hdf5_data_path, root_key='extra',
                                             cols_to_get=[trace_demo_file_name, trace_index_name, num_traces_name,
                                                          trace_one_non_replay_team_name,
                                                          trace_one_non_replay_bot_name] + trace_is_bot_player_names)

    os.makedirs(trace_humanness_path, exist_ok=True)

    for trace_index in [i for i in aggressive_trace_extra_df[trace_index_name].unique() if i != -1]:
        aggressive_one_bot_trace_round_ids = list(
            aggressive_trace_extra_df[(aggressive_trace_extra_df[trace_index_name] == trace_index) &
                                      (aggressive_trace_extra_df[trace_one_non_replay_bot_name] == 1)].index
        )
        aggressive_one_bot_metrics = get_metrics(aggressive_humanness_metrics, aggressive_one_bot_trace_round_ids,
                                                 aggressive_trace_bot_player_ids[trace_index])

        aggressive_one_team_trace_round_ids = list(
            aggressive_trace_extra_df[(aggressive_trace_extra_df[trace_index_name] == trace_index) &
                                      (aggressive_trace_extra_df[trace_one_non_replay_bot_name] == 0)].index
        )
        aggressive_one_team_metrics = get_metrics(aggressive_humanness_metrics, aggressive_one_team_trace_round_ids,
                                                  aggressive_trace_bot_player_ids[trace_index])

        passive_one_bot_trace_round_ids = list(
            passive_trace_extra_df[(passive_trace_extra_df[trace_index_name] == trace_index) &
                                   (passive_trace_extra_df[trace_one_non_replay_bot_name] == 1)].index
        )
        passive_one_bot_metrics = get_metrics(passive_humanness_metrics, passive_one_bot_trace_round_ids,
                                              passive_trace_bot_player_ids[trace_index])

        passive_one_team_trace_round_ids = list(
            passive_trace_extra_df[(passive_trace_extra_df[trace_index_name] == trace_index) &
                                   (passive_trace_extra_df[trace_one_non_replay_bot_name] == 0)].index
        )
        passive_one_team_metrics = get_metrics(passive_humanness_metrics, passive_one_team_trace_round_ids,
                                               passive_trace_bot_player_ids[trace_index])

        trace_demo_file = \
            aggressive_trace_extra_df.loc[aggressive_one_bot_trace_round_ids[0], trace_demo_file_name] \
            .decode('utf-8')[:-1]

        fig = plt.figure(figsize=(fig_length * num_player_types, fig_length * num_figs), constrained_layout=True)
        fig.suptitle("Bot in Replay Humanness Metrics " + str(trace_index) + "_" + trace_demo_file)
        axs = fig.subplots(num_figs, num_player_types, squeeze=False)

        for m in range(len(aggressive_one_bot_metrics)):
            metric_name = metric_names[m]
            plot_metric(axs, m,
                        aggressive_one_bot_metrics[metric_name], aggressive_one_team_metrics[metric_name],
                        passive_one_bot_metrics[metric_name], passive_one_team_metrics[metric_name],
                        metric_name)

        png_file_name = str(trace_index) + "_" + trace_demo_file + ".png"
        plt.savefig(trace_humanness_path / png_file_name)
        print(f"finished {png_file_name}")
