import os
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.humanness_metrics.column_names import distance_to_nearest_teammate_name, \
    distance_to_nearest_teammate_when_firing_name, distance_to_nearest_enemy_when_shot_name, \
    delta_distance_to_nearest_teammate_name, delta_distance_to_nearest_teammate_when_firing_name, \
    distance_to_c4_name, distance_to_c4_when_firing_name, distance_to_c4_when_shot_name, \
    delta_distance_to_c4_name, delta_distance_to_c4_when_firing_name, delta_distance_to_c4_when_shot_name, \
    distance_to_cover_when_enemy_visible_fov_name, distance_to_cover_when_firing_name, distance_to_cover_when_shot_name, \
    time_from_firing_to_teammate_seeing_enemy_fov_name, time_from_shot_to_teammate_seeing_enemy_fov_name, \
    distance_to_c4_when_enemy_visible_fov_name, delta_distance_to_c4_when_enemy_visible_fov_name, \
    delta_distance_to_nearest_teammate_when_shot_name, rollout_humanness_hdf5_data_path, all_train_humanness_folder_path

from learn_bot.latent.analyze.humanness_metrics.hdf5_loader import HumannessMetrics, HumannessDataOptions
from learn_bot.latent.analyze.process_trajectory_comparison import plot_hist, generate_bins, set_pd_print_options
from learn_bot.latent.analyze.test_traces.column_names import rollout_aggressive_humanness_hdf5_data_path, \
    rollout_passive_humanness_hdf5_data_path, trace_index_name, trace_one_non_replay_team_name, \
    trace_one_non_replay_bot_name, rollout_aggressive_trace_hdf5_data_path, rollout_passive_trace_hdf5_data_path, \
    trace_demo_file_name, num_traces_name, trace_is_bot_player_names, trace_humanness_path, \
    rollout_heuristic_humanness_hdf5_data_path, rollout_heuristic_trace_hdf5_data_path, \
    rollout_default_humanness_hdf5_data_path, rollout_default_trace_hdf5_data_path
from learn_bot.latent.analyze.test_traces.run_trace_creation import trace_file_name, rft_demo_file_name, rft_hdf5_key, \
    rft_ct_bot_name
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.multi_hdf5_wrapper import train_test_split_folder_path

RoundBotPlayerIds = Dict[int, List[int]]
TraceBotPlayerIds = Dict[int, RoundBotPlayerIds]

metric_cols = ['distance_to_nearest_teammate', 'distance_to_nearest_teammate_when_firing', 'distance_to_nearest_teammate_when_shot',
               'delta_distance_to_nearest_teammate', 'delta_distance_to_nearest_teammate_when_firing', 'delta_distance_to_nearest_teammate_when_shot',
               'distance_to_c4', 'distance_to_c4_when_enemy_visible_fov', 'distance_to_c4_when_firing', 'distance_to_c4_when_shot',
               'delta_distance_to_c4', 'delta_distance_to_c4_when_enemy_visible_fov', 'delta_distance_to_c4_when_firing', 'delta_distance_to_c4_when_shot',
               'distance_to_cover_when_enemy_visible_fov', 'distance_to_cover_when_firing', 'distance_to_cover_when_shot',
               'time_from_firing_to_teammate_seeing_enemy_fov', 'time_from_shot_to_teammate_seeing_enemy_fov']

metric_names = [distance_to_nearest_teammate_name, distance_to_nearest_teammate_when_firing_name, distance_to_nearest_enemy_when_shot_name,
                delta_distance_to_nearest_teammate_name, delta_distance_to_nearest_teammate_when_firing_name, delta_distance_to_nearest_teammate_when_shot_name,
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

is_ct_cols = ['is_ct_per_nearest_teammate', 'is_ct_per_nearest_teammate_firing', 'is_ct_per_nearest_teammate_shot',
              'is_ct_per_nearest_teammate', 'is_ct_per_nearest_teammate_firing', 'is_ct_per_nearest_teammate_shot',
              'is_ct_per_pat', 'is_ct_per_enemy_visible_fov_pat', 'is_ct_per_firing_pat', 'is_ct_per_shot_pat',
              'is_ct_per_pat', 'is_ct_per_enemy_visible_fov_pat', 'is_ct_per_firing_pat', 'is_ct_per_shot_pat',
              'is_ct_per_enemy_visible_fov_pat', 'is_ct_per_firing_pat', 'is_ct_per_shot_pat',
              'is_ct_per_firing_to_teammate_seeing_enemy', 'is_ct_per_shot_to_teammate_seeing_enemy']

player_id_cols = ['player_id_per_nearest_teammate', 'player_id_per_nearest_teammate_firing', 'player_id_per_nearest_teammate_shot',
                  'player_id_per_nearest_teammate', 'player_id_per_nearest_teammate_firing', 'player_id_per_nearest_teammate_shot',
                  'player_id_per_pat', 'player_id_per_enemy_visible_fov_pat', 'player_id_per_firing_pat', 'player_id_per_shot_pat',
                  'player_id_per_pat', 'player_id_per_enemy_visible_fov_pat', 'player_id_per_firing_pat', 'player_id_per_shot_pat',
                  'player_id_per_enemy_visible_fov_pat', 'player_id_per_firing_pat', 'player_id_per_shot_pat',
                  'player_id_per_firing_to_teammate_seeing_enemy', 'player_id_per_shot_to_teammate_seeing_enemy']

fig_length = 8
num_figs = len(metric_cols)
num_player_types = 9
one_bot_aggressive_learned_bot_name = "One Bot Aggressive "
one_team_aggressive_learned_bot_name = "One Team Aggressive "
one_bot_passive_learned_bot_name = "One Bot Passive "
one_team_passive_learned_bot_name = "One Team Passive "
one_bot_heuristic_learned_bot_name = "One Bot Heuristic "
one_team_heuristic_learned_bot_name = "One Team Heuristic "
one_bot_default_learned_bot_name = "One Bot Default "
one_team_default_learned_bot_name = "One Team Default "
human_name = "Human "
player_type_names = [
    one_bot_aggressive_learned_bot_name,
    one_team_aggressive_learned_bot_name,
    one_bot_passive_learned_bot_name,
    one_team_passive_learned_bot_name,
    one_bot_heuristic_learned_bot_name,
    one_team_heuristic_learned_bot_name,
    one_bot_default_learned_bot_name,
    one_team_default_learned_bot_name,
    human_name]

trace_metrics: Set[Tuple[int, str]] = {(0, delta_distance_to_nearest_teammate_name), (0, delta_distance_to_c4_name),
                                       (1, delta_distance_to_nearest_teammate_name), (1, delta_distance_to_c4_name),
                                       (2, delta_distance_to_nearest_teammate_name), (2, delta_distance_to_c4_name),
                                       (3, delta_distance_to_nearest_teammate_name), (3, delta_distance_to_c4_name),
                                       (4, distance_to_cover_when_firing_name), (5, distance_to_cover_when_firing_name),
                                       (6, time_from_firing_to_teammate_seeing_enemy_fov_name), (7, time_from_firing_to_teammate_seeing_enemy_fov_name),
                                       (8, time_from_firing_to_teammate_seeing_enemy_fov_name), (9, time_from_firing_to_teammate_seeing_enemy_fov_name),
                                       }

def plot_metric(axs, metric_index: int, 
                metric_player_types: List[np.ndarray],
                metric_name: str, trace_index: int):
    values_for_max = []
    values_for_min = []
    for metric_player_type in metric_player_types:
        if len(metric_player_type) > 0:
            values_for_max.append(metric_player_type.max())
            values_for_min.append(metric_player_type.min())
    if len(values_for_max) == 0:
        return
    max_value = int(ceil(max(values_for_max)))
    if max_value == 0:
        return
    for i, player_type_name in enumerate(player_type_names):
        axs[metric_index, i].set_title(player_type_name + metric_name)

    min_value = int(floor(min(values_for_min)))
    if min_value > 0:
        min_value = 0
    bin_width = (max_value - min_value) // 20
    if bin_width == 0:
        bin_width = 1
    bins = generate_bins(min_value, max_value, bin_width)

    player_type_names_to_print = []
    avgs = []
    for i, metric_player_type in enumerate(metric_player_types):
        if len(metric_player_type) > 0:
            metric_series = pd.Series(metric_player_type)
            plot_hist(axs[metric_index, i], metric_series, bins)
            axs[metric_index, i].text((min_value + max_value) / 2., 0.4, metric_series.describe().to_string(),
                                      family='monospace')
            if 'Bot' not in player_type_names[i]:
                player_type_names_to_print.append(player_type_names[i])
                avgs.append(f"{metric_series.mean():.1f}")

    if (trace_index, metric_name) in trace_metrics:
        print(f"{trace_index}, {metric_name}")
        print(player_type_names_to_print)
        print(" & ".join(avgs))

    for i in range(num_player_types):
        axs[metric_index, i].set_ylim(0., 1.)
        axs[metric_index, i].set_xlim(min_value, max_value)


@dataclass
class HumannessDataForHumanRound:
    humanness_metrics: HumannessMetrics
    round_id: int
    ct_bot: bool


def get_metrics(humanness_metrics: HumannessMetrics, round_ids: List[int], bot_player_ids: RoundBotPlayerIds,
                ct_bot: Optional[bool] = None) \
        -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for i in range(len(metric_cols)):
        metric_col = humanness_metrics.__getattribute__(metric_cols[i])
        round_ids_col = humanness_metrics.__getattribute__(round_id_cols[i])
        player_ids_col = humanness_metrics.__getattribute__(player_id_cols[i])
        is_ct_col = humanness_metrics.__getattribute__(is_ct_cols[i])
        # start with false, because building up or's
        conditions = round_ids_col == -1
        for round_id in round_ids:
            if ct_bot is None:
                conditions = conditions | \
                             ((round_ids_col == round_id) & (np.isin(player_ids_col, bot_player_ids[round_id])))
            else:
                conditions = conditions | ((round_ids_col == round_id) & (is_ct_col == ct_bot))
        result[metric_names[i]] = metric_col[conditions]
    return result


def get_humanness_data_for_human_round(trace_extra_df: pd.DataFrame, trace_index: int) -> HumannessDataForHumanRound:
    trace_extra_row = trace_extra_df.iloc[trace_index]
    round_id = trace_extra_row[round_id_column]
    trace_hdf5_key = trace_extra_row[rft_hdf5_key].decode('utf-8')
    humanness_hdf5_key = trace_hdf5_key.replace('behaviorTreeTeamFeatureStore', 'humannessMetrics')
    ct_bot = trace_extra_row[rft_ct_bot_name]

    human_humanness_metrics = HumannessMetrics(HumannessDataOptions.CUSTOM, False,
                                               all_train_humanness_folder_path.parent / humanness_hdf5_key)
    return HumannessDataForHumanRound(human_humanness_metrics, round_id, ct_bot)


def plot_humanness_metrics(aggressive_trace_bot_player_ids: TraceBotPlayerIds,
                           passive_trace_bot_player_ids: TraceBotPlayerIds,
                           heuristic_trace_bot_player_ids: TraceBotPlayerIds,
                           default_trace_bot_player_ids: TraceBotPlayerIds):
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
    heuristic_humanness_metrics = HumannessMetrics(HumannessDataOptions.CUSTOM, False,
                                                   rollout_heuristic_humanness_hdf5_data_path)
    heuristic_trace_extra_df = load_hdf5_to_pd(rollout_heuristic_trace_hdf5_data_path, root_key='extra',
                                               cols_to_get=[trace_demo_file_name, trace_index_name, num_traces_name,
                                                            trace_one_non_replay_team_name,
                                                            trace_one_non_replay_bot_name] + trace_is_bot_player_names)
    default_humanness_metrics = HumannessMetrics(HumannessDataOptions.CUSTOM, False,
                                                 rollout_default_humanness_hdf5_data_path)
    default_trace_extra_df = load_hdf5_to_pd(rollout_default_trace_hdf5_data_path, root_key='extra',
                                             cols_to_get=[trace_demo_file_name, trace_index_name, num_traces_name,
                                                          trace_one_non_replay_team_name,
                                                          trace_one_non_replay_bot_name] + trace_is_bot_player_names)

    trace_path = train_test_split_folder_path / trace_file_name
    trace_extra_df = load_hdf5_to_pd(trace_path, root_key='extra', cols_to_get=[rft_demo_file_name, round_id_column,
                                                                                rft_hdf5_key, rft_ct_bot_name])

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

        heuristic_one_bot_trace_round_ids = list(
            heuristic_trace_extra_df[(heuristic_trace_extra_df[trace_index_name] == trace_index) &
                                     (heuristic_trace_extra_df[trace_one_non_replay_bot_name] == 1)].index
        )
        heuristic_one_bot_metrics = get_metrics(heuristic_humanness_metrics, heuristic_one_bot_trace_round_ids,
                                                 heuristic_trace_bot_player_ids[trace_index])
        heuristic_one_team_trace_round_ids = list(
            heuristic_trace_extra_df[(heuristic_trace_extra_df[trace_index_name] == trace_index) &
                                     (heuristic_trace_extra_df[trace_one_non_replay_bot_name] == 0)].index
        )
        heuristic_one_team_metrics = get_metrics(heuristic_humanness_metrics, heuristic_one_team_trace_round_ids,
                                                 heuristic_trace_bot_player_ids[trace_index])

        default_one_bot_trace_round_ids = list(
            default_trace_extra_df[(default_trace_extra_df[trace_index_name] == trace_index) &
                                   (default_trace_extra_df[trace_one_non_replay_bot_name] == 1)].index
        )
        default_one_bot_metrics = get_metrics(default_humanness_metrics, default_one_bot_trace_round_ids,
                                               default_trace_bot_player_ids[trace_index])
        default_one_team_trace_round_ids = list(
            default_trace_extra_df[(default_trace_extra_df[trace_index_name] == trace_index) &
                                   (default_trace_extra_df[trace_one_non_replay_bot_name] == 0)].index
        )
        default_one_team_metrics = get_metrics(default_humanness_metrics, default_one_team_trace_round_ids,
                                               default_trace_bot_player_ids[trace_index])

        humanness_data_for_human_round = get_humanness_data_for_human_round(trace_extra_df, trace_index)
        human_metrics = get_metrics(humanness_data_for_human_round.humanness_metrics,
                                    [humanness_data_for_human_round.round_id],
                                    {}, humanness_data_for_human_round.ct_bot)


        trace_demo_file = \
            aggressive_trace_extra_df.loc[aggressive_one_bot_trace_round_ids[0], trace_demo_file_name] \
            .decode('utf-8')[:-1]

        fig = plt.figure(figsize=(fig_length * num_player_types, fig_length * num_figs), constrained_layout=True)
        fig.suptitle("Bot in Replay Humanness Metrics " + str(trace_index) + "_" + trace_demo_file)
        axs = fig.subplots(num_figs, num_player_types, squeeze=False)

        for m in range(len(aggressive_one_bot_metrics)):
            metric_name = metric_names[m]
            plot_metric(axs, m,
                        [
                            aggressive_one_bot_metrics[metric_name],
                            aggressive_one_team_metrics[metric_name],
                            passive_one_bot_metrics[metric_name],
                            passive_one_team_metrics[metric_name],
                            heuristic_one_bot_metrics[metric_name],
                            heuristic_one_team_metrics[metric_name],
                            default_one_bot_metrics[metric_name],
                            default_one_team_metrics[metric_name],
                            human_metrics[metric_name]
                        ], metric_name, trace_index)

        png_file_name = str(trace_index) + "_" + trace_demo_file + ".png"
        plt.savefig(trace_humanness_path / png_file_name)
        print(f"finished {png_file_name}")
