from math import ceil
from pathlib import Path
from pprint import pformat
from typing import List, Dict, Union, Optional

import pandas as pd
import scipy
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import plot_hist, generate_bins
from learn_bot.latent.analyze.plot_trajectory_heatmap.render_key_places_and_mistakes import plot_specific_key_places, \
    plot_key_places, plot_mistakes
from learn_bot.latent.analyze.plot_trajectory_heatmap.render_time_to_event_and_distance_histograms import \
    compute_time_to_event_and_distance_histograms
from learn_bot.latent.analyze.plot_trajectory_heatmap.title_rename_dict import title_rename_dict
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import get_title_to_speeds, \
    get_title_to_lifetimes, get_title_to_shots_per_kill, get_title_to_ttk_and_distance, time_to_kill_col, \
    get_title_to_tts_and_distance, time_to_shoot_col, get_title_to_tts_and_distance_time_constrained, \
    get_title_to_ttk_and_distance_time_constrained, vel_col, get_title_to_delta_speeds, \
    get_title_to_action_changes, title_to_action_changes_when_killing, get_title_to_action_changes_when_killing, \
    get_title_to_action_changes_when_shooting

fig_length = 6


def compute_one_metric_histograms(title_to_values: Dict[str, List[float]], metric_title: str,
                                  bin_width: Union[int, float], smallest_max: float,
                                  x_label: str, plot_file_path: Path, add_points_per_bin: bool = False):
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
        plot_hist(axs[ax_index, 0], pd.Series(values), bins, add_points_per_bin)
        axs[ax_index, 0].set_xlim(0., max_observed)
        axs[ax_index, 0].set_ylim(0., 1.)
        axs[ax_index, 0].set_title(title + " " + metric_title)
        axs[ax_index, 0].set_xlabel(x_label)
        ax_index += 1
    plt.savefig(plot_file_path)


plt.rc('font', family='Arial')


def compute_one_metric_four_histograms(title_to_values: Dict[str, List[float]], metric_title: Optional[str],
                                       bin_width: Union[int, float], max_bin_end: float, y_max: float,
                                       x_label: str, x_ticks: List, y_label: Optional[str],
                                       y_ticks: List, plot_file_path: Path):
    if len(title_to_values) != 4:
        return
    local_fig_length = 3.3
    fig = plt.figure(figsize=(local_fig_length, local_fig_length / 3), constrained_layout=True)
    axs = fig.subplots(1, 4, squeeze=False, sharey=True)

    if y_label is not None:
        if metric_title is not None:
            fig.suptitle(metric_title, x=0.52, fontsize=8)
        fig.supxlabel(x_label, x=0.52, fontsize=8)
        fig.supylabel(y_label, fontsize=8)
    else:
        if metric_title is not None:
            fig.suptitle(metric_title, fontsize=8)
        fig.supxlabel(x_label, fontsize=8)

    if bin_width < 1.:
        num_bins = int(ceil(max_bin_end / bin_width))
        # add 1 as need left edge of every bin and right edge of last bin
        bins = [i * bin_width for i in range(num_bins + 1)]
    else:
        bins = generate_bins(0, int(ceil(max_bin_end)), bin_width)
    ax_index = 0
    for title, values in title_to_values.items():
        renamed_title = title_rename_dict[title]
        row_index = 0
        col_index = ax_index
        ax = axs[row_index, col_index]
        plot_hist(axs[row_index, col_index], pd.Series(values), bins)
        ax.set_xlim(0., max_bin_end)
        ax.set_ylim(0., y_max)
        ax.set_title(renamed_title, fontsize=8)
        #ax.set_xlabel(x_label)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="x", labelsize=8, pad=1)
        ax.tick_params(axis="y", labelsize=8, pad=1)

        # remove right/top spine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # remove veritcal grid lines, make horizontal dotted
        ax.yaxis.grid(False)#True, color='#EEEEEE', dashes=[4,1])
        ax.xaxis.grid(False)

        ax_index += 1
    plt.savefig(plot_file_path)


def compute_one_metric_emd(title_to_values: Dict[str, List[float]], plot_file_path: Path):
    if 'Human' not in title_to_values.keys():
        return
    non_human_titles = [t for t in title_to_values.keys() if t != 'Human']
    title_to_emd: Dict[str, float] = {}
    for t in non_human_titles:
        title_to_emd[t] = scipy.stats.wasserstein_distance(title_to_values['Human'], title_to_values[t])
    with open(plot_file_path, 'w') as f:
        f.write(pformat(title_to_emd, indent=4))


def compute_metrics(trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    if trajectory_filter_options.is_no_filter():
        #plot_most_common_places_by_title(plots_path / 'most_popular_places.png')
        #key_places_by_title = get_key_places_by_title()
        #key_places_by_title.plot(kind='bar', rot=90, title='Rounds With Team Formations')
        #plt.savefig(plots_path / 'key_places.png', bbox_inches='tight')
        plot_key_places(plots_path, True)
        plot_key_places(plots_path, False)
        plot_specific_key_places(plots_path)
        plot_mistakes(plots_path, True)
        plot_mistakes(plots_path, False)


    if trajectory_filter_options.compute_speeds:
        # airstrafing can get you above normal weapon max speed
        compute_one_metric_histograms(get_title_to_speeds(), 'Weapon/Scoped Scaled Speed', 0.1, 1., 'Percent Max Speed',
                                      plots_path / ('speeds_' + str(trajectory_filter_options) + '.png'))
    if trajectory_filter_options.compute_action_changes:
        compute_one_metric_histograms(get_title_to_delta_speeds(), 'Weapon/Scoped Scaled Delta Speed', 0.02, 1.,
                                      'Percent Max Speed',
                                      plots_path / ('delta_speeds_' + str(trajectory_filter_options) + '.png'))
        compute_one_metric_histograms(get_title_to_action_changes(), 'Move/NotMove Changes', 1, 4,
                                      'Percent Changes',
                                      plots_path / ('move_not_move_' + str(trajectory_filter_options) + '.png'),
                                      add_points_per_bin=True)
        compute_one_metric_histograms(get_title_to_action_changes_when_shooting(), 'Move/NotMove Changes When Shoot 2s', 1, 4,
                                      'Percent Changes',
                                      plots_path / ('move_not_move_shoot_2s' + str(trajectory_filter_options) + '.png'),
                                      add_points_per_bin=True)
        compute_one_metric_histograms(get_title_to_action_changes_when_killing(), 'Move/NotMove Changes When Kill 2s', 1, 4,
                                      'Percent Changes',
                                      plots_path / ('move_not_move_kill_2s' + str(trajectory_filter_options) + '.png'),
                                      add_points_per_bin=True)
    if trajectory_filter_options.compute_lifetimes:
        # small timing mismatch can get 41 seconds on bomb timer
        compute_one_metric_four_histograms(get_title_to_lifetimes(), None, 5, 40.,
                                           0.6, 'Player Lifetimes (Seconds)', [0, 20, 40], None, [0, 0.3, 0.6],
                                           plots_path / ('lifetimes_' + str(trajectory_filter_options) + '.pdf'))
        compute_one_metric_emd(get_title_to_lifetimes(),
                               plots_path / ('lifetimes_' + str(trajectory_filter_options) + '.txt'))
    if trajectory_filter_options.compute_shots_per_kill:
        compute_one_metric_four_histograms(get_title_to_shots_per_kill(), None, 1, 30.,
                                           0.3, 'Shots Per Kill', [0, 15, 30], None, [0, 0.15, 0.3],
                                           plots_path / ('shots_per_kill_' + str(trajectory_filter_options) + '.pdf'))
        compute_one_metric_emd(get_title_to_shots_per_kill(),
                               plots_path / ('shots_per_kill_' + str(trajectory_filter_options) + '.txt'))
    if trajectory_filter_options.compute_crosshair_distance_to_engage:
        compute_time_to_event_and_distance_histograms(get_title_to_tts_and_distance(), 'Time To Shoot', time_to_shoot_col,
                                                      'Seconds', plots_path)
        compute_time_to_event_and_distance_histograms(get_title_to_tts_and_distance_time_constrained(), 'Time To Shoot 2s Constraint', time_to_shoot_col,
                                                      'Seconds', plots_path)
        compute_time_to_event_and_distance_histograms(get_title_to_ttk_and_distance(), 'Time To Kill', time_to_kill_col,
                                                      'Seconds', plots_path)
        compute_time_to_event_and_distance_histograms(get_title_to_ttk_and_distance_time_constrained(), 'Time To Kill 2s Constraint', time_to_kill_col,
                                                      'Seconds', plots_path)
        compute_time_to_event_and_distance_histograms(get_title_to_tts_and_distance_time_constrained(), 'Speed When Shoot 2s Constraint', vel_col,
                                                      'Speed', plots_path)
        compute_time_to_event_and_distance_histograms(get_title_to_ttk_and_distance_time_constrained(), 'Speed When Kill 2s Constraint', vel_col,
                                                      'Speed', plots_path)
