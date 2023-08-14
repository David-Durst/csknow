import os
from math import ceil

import pandas as pd

from learn_bot.latent.analyze.humanness_metrics.hdf5_loader import *
from learn_bot.latent.analyze.process_trajectory_comparison import set_pd_print_options, generate_bins, plot_hist
import matplotlib.pyplot as plt

fig_length = 6
num_figs = 26
num_player_types = 4
learned_bot_name = "Learned Bot "
human_name = "Human "
heuristic_bot_name = "Hand-Crafted Bot "
default_bot_name = "Default Bot "


def plot_metric(axs, metric_index: int, human_metric: np.ndarray, learned_bot_metric: np.ndarray,
                heuristic_bot_metric: np.ndarray, default_bot_metric: np.ndarray,
                metric_name: str, pct_bins: bool = False):
    max_value = int(ceil(max(human_metric.max(), learned_bot_metric.max(),
                             heuristic_bot_metric.max(), default_bot_metric.max())))
    axs[metric_index, 0].set_title(learned_bot_name + metric_name)
    axs[metric_index, 1].set_title(human_name + metric_name)
    axs[metric_index, 2].set_title(heuristic_bot_name + metric_name)
    axs[metric_index, 3].set_title(default_bot_name + metric_name)
    bins: List
    if pct_bins:
        bins = [i * 0.1 for i in range(11)]
    else:
        bins = generate_bins(0, max_value, max_value // 20)
    plot_hist(axs[metric_index, 0], pd.Series(learned_bot_metric), bins)
    plot_hist(axs[metric_index, 1], pd.Series(human_metric), bins)
    plot_hist(axs[metric_index, 2], pd.Series(heuristic_bot_metric), bins)
    plot_hist(axs[metric_index, 3], pd.Series(default_bot_metric), bins)
    axs[metric_index, 0].set_ylim(0., 1.)
    axs[metric_index, 1].set_ylim(0., 1.)
    axs[metric_index, 2].set_ylim(0., 1.)
    axs[metric_index, 3].set_ylim(0., 1.)


def plot_sums(axs, metric_index: int, human_metric: np.ndarray, learned_bot_metric: np.ndarray,
              heuristic_bot_metric: np.ndarray, default_bot_metric: np.ndarray, metric_name: str):
    axs[metric_index, 0].set_title(metric_name)
    learned_bot_value = learned_bot_metric.sum() / len(learned_bot_metric)
    human_value = human_metric.sum() / len(human_metric)
    heuristic_bot_value = heuristic_bot_metric.sum() / len(heuristic_bot_metric)
    default_bot_value = default_bot_metric.sum() / len(default_bot_metric)
    players = [learned_bot_name + metric_name, human_name + metric_name, heuristic_bot_name + metric_name,
               default_bot_name + metric_name]
    values = [learned_bot_value, human_value, heuristic_bot_value, default_bot_value]
    axs[metric_index, 0].bar(players, values)
    axs[metric_index, 0].set_ylim(0., 1.)


def run_humanness():
    all_train_humanness_metrics = HumannessMetrics(HumannessDataOptions.ALL_TRAIN)
    rollout_humanness_metrics = HumannessMetrics(HumannessDataOptions.ROLLOUT)
    heuristics_humanness_metrics = HumannessMetrics(HumannessDataOptions.HEURISTIC)
    default_humanness_metrics = HumannessMetrics(HumannessDataOptions.DEFAULT)
    print(f"max distance to nearest enemy human {all_train_humanness_metrics.distance_to_nearest_enemy.max()}, "
          f"learned {rollout_humanness_metrics.distance_to_nearest_enemy.max()}, "
          f"hand-crafted {heuristics_humanness_metrics.distance_to_nearest_enemy.max()}, "
          f"default {default_humanness_metrics.distance_to_nearest_enemy.max()}")

    set_pd_print_options()


    fig = plt.figure(figsize=(fig_length*num_player_types, fig_length*num_figs), constrained_layout=True)
    fig.suptitle("Learned Bot vs Human vs Hand-Crafted Bot vs Default Bot Metrics")
    axs = fig.subplots(num_figs, num_player_types, squeeze=False)

    plot_metric(axs, 0,
                all_train_humanness_metrics.unscaled_speed,
                rollout_humanness_metrics.unscaled_speed,
                heuristics_humanness_metrics.unscaled_speed,
                default_humanness_metrics.unscaled_speed,
                unscaled_speed_name)
    plot_metric(axs, 1,
                all_train_humanness_metrics.unscaled_speed_when_firing,
                rollout_humanness_metrics.unscaled_speed_when_firing,
                heuristics_humanness_metrics.unscaled_speed_when_firing,
                default_humanness_metrics.unscaled_speed_when_firing,
                unscaled_speed_when_firing_name)
    plot_metric(axs, 2,
                all_train_humanness_metrics.unscaled_speed_when_shot,
                rollout_humanness_metrics.unscaled_speed_when_shot,
                heuristics_humanness_metrics.unscaled_speed_when_shot,
                default_humanness_metrics.unscaled_speed_when_shot,
                unscaled_speed_when_shot_name)

    plot_metric(axs, 3,
                all_train_humanness_metrics.scaled_speed,
                rollout_humanness_metrics.scaled_speed,
                heuristics_humanness_metrics.scaled_speed,
                default_humanness_metrics.scaled_speed,
                scaled_speed_name, pct_bins=True)
    plot_metric(axs, 4,
                all_train_humanness_metrics.scaled_speed_when_firing,
                rollout_humanness_metrics.scaled_speed_when_firing,
                heuristics_humanness_metrics.scaled_speed_when_firing,
                default_humanness_metrics.scaled_speed_when_firing,
                scaled_speed_when_firing_name, pct_bins=True)
    plot_metric(axs, 5,
                all_train_humanness_metrics.scaled_speed_when_shot,
                rollout_humanness_metrics.scaled_speed_when_shot,
                heuristics_humanness_metrics.scaled_speed_when_shot,
                default_humanness_metrics.scaled_speed_when_shot,
                scaled_speed_when_shot_name, pct_bins=True)

    plot_metric(axs, 6,
                all_train_humanness_metrics.weapon_only_scaled_speed,
                rollout_humanness_metrics.weapon_only_scaled_speed,
                heuristics_humanness_metrics.weapon_only_scaled_speed,
                default_humanness_metrics.weapon_only_scaled_speed,
                weapon_only_scaled_speed_name, pct_bins=True)
    plot_metric(axs, 7,
                all_train_humanness_metrics.weapon_only_scaled_speed_when_firing,
                rollout_humanness_metrics.weapon_only_scaled_speed_when_firing,
                heuristics_humanness_metrics.weapon_only_scaled_speed_when_firing,
                default_humanness_metrics.weapon_only_scaled_speed_when_firing,
                weapon_only_scaled_speed_when_firing_name, pct_bins=True)
    plot_metric(axs, 8,
                all_train_humanness_metrics.weapon_only_scaled_speed_when_shot,
                rollout_humanness_metrics.weapon_only_scaled_speed_when_shot,
                heuristics_humanness_metrics.weapon_only_scaled_speed_when_shot,
                default_humanness_metrics.weapon_only_scaled_speed_when_shot,
                weapon_only_scaled_speed_when_shot_name, pct_bins=True)

    plot_metric(axs, 9,
                all_train_humanness_metrics.distance_to_nearest_teammate,
                rollout_humanness_metrics.distance_to_nearest_teammate,
                heuristics_humanness_metrics.distance_to_nearest_teammate,
                default_humanness_metrics.distance_to_nearest_teammate,
                distance_to_nearest_teammate_name)
    plot_metric(axs, 10,
                all_train_humanness_metrics.distance_to_nearest_teammate_when_firing,
                rollout_humanness_metrics.distance_to_nearest_teammate_when_firing,
                heuristics_humanness_metrics.distance_to_nearest_teammate_when_firing,
                default_humanness_metrics.distance_to_nearest_teammate_when_firing,
                distance_to_nearest_teammate_when_firing_name)
    plot_metric(axs, 11,
                all_train_humanness_metrics.distance_to_nearest_teammate_when_shot,
                rollout_humanness_metrics.distance_to_nearest_teammate_when_shot,
                heuristics_humanness_metrics.distance_to_nearest_teammate_when_shot,
                default_humanness_metrics.distance_to_nearest_teammate_when_shot,
                distance_to_nearest_teammate_when_shot_name)

    plot_metric(axs, 12,
                all_train_humanness_metrics.distance_to_nearest_enemy,
                rollout_humanness_metrics.distance_to_nearest_enemy,
                heuristics_humanness_metrics.distance_to_nearest_enemy,
                default_humanness_metrics.distance_to_nearest_enemy,
                distance_to_nearest_enemy_name)
    plot_metric(axs, 13,
                all_train_humanness_metrics.distance_to_nearest_enemy_when_firing,
                rollout_humanness_metrics.distance_to_nearest_enemy_when_firing,
                heuristics_humanness_metrics.distance_to_nearest_enemy_when_firing,
                default_humanness_metrics.distance_to_nearest_enemy_when_firing,
                distance_to_nearest_enemy_when_firing_name)
    plot_metric(axs, 14,
                all_train_humanness_metrics.distance_to_nearest_enemy_when_shot,
                rollout_humanness_metrics.distance_to_nearest_enemy_when_shot,
                heuristics_humanness_metrics.distance_to_nearest_enemy_when_shot,
                default_humanness_metrics.distance_to_nearest_enemy_when_shot,
                distance_to_nearest_enemy_when_shot_name)

    plot_metric(axs, 15,
                all_train_humanness_metrics.distance_to_attacker_when_shot,
                rollout_humanness_metrics.distance_to_attacker_when_shot,
                heuristics_humanness_metrics.distance_to_attacker_when_shot,
                default_humanness_metrics.distance_to_attacker_when_shot,
                distance_to_attacker_when_shot_name)

    plot_metric(axs, 16,
                all_train_humanness_metrics.distance_to_cover,
                rollout_humanness_metrics.distance_to_cover,
                heuristics_humanness_metrics.distance_to_cover,
                default_humanness_metrics.distance_to_cover,
                distance_to_cover_name)
    plot_metric(axs, 17,
                all_train_humanness_metrics.distance_to_cover_when_enemy_visible_no_fov,
                rollout_humanness_metrics.distance_to_cover_when_enemy_visible_no_fov,
                heuristics_humanness_metrics.distance_to_cover_when_enemy_visible_no_fov,
                default_humanness_metrics.distance_to_cover_when_enemy_visible_no_fov,
                distance_to_cover_when_enemy_visible_no_fov_name)
    plot_metric(axs, 18,
                all_train_humanness_metrics.distance_to_cover_when_enemy_visible_fov,
                rollout_humanness_metrics.distance_to_cover_when_enemy_visible_fov,
                heuristics_humanness_metrics.distance_to_cover_when_enemy_visible_fov,
                default_humanness_metrics.distance_to_cover_when_enemy_visible_fov,
                distance_to_cover_when_enemy_visible_fov_name)
    plot_metric(axs, 19,
                all_train_humanness_metrics.distance_to_cover_when_firing,
                rollout_humanness_metrics.distance_to_cover_when_firing,
                heuristics_humanness_metrics.distance_to_cover_when_firing,
                default_humanness_metrics.distance_to_cover_when_firing,
                distance_to_cover_when_firing_name)
    plot_metric(axs, 20,
                all_train_humanness_metrics.distance_to_cover_when_shot,
                rollout_humanness_metrics.distance_to_cover_when_shot,
                heuristics_humanness_metrics.distance_to_cover_when_shot,
                default_humanness_metrics.distance_to_cover_when_shot,
                distance_to_cover_when_shot_name)

    plot_metric(axs, 21,
                all_train_humanness_metrics.pct_time_max_speed_ct,
                rollout_humanness_metrics.pct_time_max_speed_ct,
                heuristics_humanness_metrics.pct_time_max_speed_ct,
                default_humanness_metrics.pct_time_max_speed_ct,
                pct_time_max_speed_ct_name, pct_bins=True)
    plot_metric(axs, 22,
                all_train_humanness_metrics.pct_time_max_speed_t,
                rollout_humanness_metrics.pct_time_max_speed_t,
                heuristics_humanness_metrics.pct_time_max_speed_t,
                default_humanness_metrics.pct_time_max_speed_t,
                pct_time_max_speed_t_name, pct_bins=True)
    plot_metric(axs, 23,
                all_train_humanness_metrics.pct_time_still_ct,
                rollout_humanness_metrics.pct_time_still_ct,
                heuristics_humanness_metrics.pct_time_still_ct,
                default_humanness_metrics.pct_time_still_ct,
                pct_time_still_ct_name, pct_bins=True)
    plot_metric(axs, 24,
                all_train_humanness_metrics.pct_time_still_t,
                rollout_humanness_metrics.pct_time_still_t,
                heuristics_humanness_metrics.pct_time_still_t,
                default_humanness_metrics.pct_time_still_t,
                pct_time_still_t_name, pct_bins=True)
    plot_sums(axs, 25,
              all_train_humanness_metrics.ct_wins,
              rollout_humanness_metrics.ct_wins,
              heuristics_humanness_metrics.ct_wins,
              default_humanness_metrics.ct_wins,
              ct_wins_name)

    os.makedirs(humanness_plots_path, exist_ok=True)
    plt.savefig(humanness_plots_path / 'humanness_metrics.png')

if __name__ == "__main__":
    run_humanness()