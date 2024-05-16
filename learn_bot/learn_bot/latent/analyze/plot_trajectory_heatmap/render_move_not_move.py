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
    get_title_to_action_changes_when_shooting, get_title_to_num_points, get_title_to_action_changes_when_enemy_visible

def compute_move_not_move_histograms(title_to_values: Dict[str, List[float]], metric_title: str,
                                    bin_width: Union[int, float], smallest_max: float,
                                    plot_file_path: Path,
                                    fuse_by_state_changing: bool = False, fuse_not_changing_only: bool = False):
    num_titles = len(title_to_values.keys())
    fig = plt.figure(figsize=(3.5, 3.3 * num_titles))
    axs = fig.subplots(num_titles, 1, squeeze=False)

    # get the max value for histogram (if bigger than smaller max), min wil be 0 always
    max_observed = smallest_max
    for title, values in title_to_values.items():
        max_observed = max(max_observed, max(values))

    if fuse_by_state_changing:
        max_observed = min(2, max_observed)
    elif fuse_not_changing_only:
        max_observed = min(3, max_observed)

    if bin_width < 1.:
        num_bins = int(ceil(max_observed / bin_width))
        # add 1 as need left edge of every bin and right edge of last bin
        bins = [i * bin_width for i in range(num_bins + 1)]
    else:
        bins = generate_bins(0, int(ceil(max_observed)), bin_width)
    ax_index = 0
    for title, values in title_to_values.items():
        renamed_title = title_rename_dict[title]
        values_series = pd.Series(values)
        if fuse_by_state_changing:
            x_ticks = [0.5, 1.5]
            x_ticklabels = ["Repeat Action", "Change Action"]
            values_series[values_series < 2] = 0
            values_series[values_series >= 2] = 1
        elif fuse_not_changing_only:
            x_ticks = [0.5, 1.5, 2.5]
            x_ticklabels = ["Stay Still", "Keep Moving", "Change Action"]
            values_series[values_series >= 2] = 2
        else:
            x_ticks = [0.5, 1.5, 2.5, 3.5]
            x_ticklabels = ["Stay Still", "Keep Moving", "Start Moving", "Stop Moving"]
        plot_hist(axs[ax_index, 0], values_series, bins)
        axs[ax_index, 0].set_xlim(0., max_observed)
        axs[ax_index, 0].set_ylim(0., 1.)
        axs[ax_index, 0].set_title(renamed_title + " " + metric_title, fontsize=8)
        axs[ax_index, 0].set_ylabel("Percent of Ticks", fontsize=8)
        axs[ax_index, 0].set_xticks(x_ticks)
        axs[ax_index, 0].tick_params(axis="x", labelsize=8)
        axs[ax_index, 0].tick_params(axis="y", labelsize=8)
        axs[ax_index, 0].set_xticklabels(x_ticklabels)
        axs[ax_index, 0].set_yticks([0, 0.25, 0.5, 0.75, 1.])
        axs[ax_index, 0].grid(visible=False)
        #axs[ax_index, 0].yaxis.grid(True, color='#EEEEEE', dashes=[4, 1])
        #axs[ax_index, 0].xaxis.grid(False)
        axs[ax_index, 0].spines['top'].set_visible(False)
        axs[ax_index, 0].spines['right'].set_visible(False)
        ax_index += 1
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.2)
    plt.savefig(plot_file_path, bbox_inches='tight')


def render_all_move_not_move_histograms(trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    compute_move_not_move_histograms(get_title_to_action_changes(), 'Move State', 1, 4,
                                     plots_path / ('specific_move_not_move_' + str(trajectory_filter_options) + '.pdf'))
    compute_move_not_move_histograms(get_title_to_action_changes(), 'Move State', 1, 4,
                                     plots_path / ('specific_3_cat_move_not_move_' + str(trajectory_filter_options) + '.pdf'),
                                     fuse_not_changing_only=True)
    compute_move_not_move_histograms(get_title_to_action_changes(), 'Move State', 1, 4,
                                     plots_path / ('specific_2_cat_move_not_move_' + str(trajectory_filter_options) + '.pdf'),
                                     fuse_by_state_changing=True)

    compute_move_not_move_histograms(get_title_to_action_changes_when_shooting(), 'Shooting Move State', 1, 4,
                                      plots_path / ('specific_move_not_move_shoot_2s' + str(trajectory_filter_options) + '.pdf'))
    compute_move_not_move_histograms(get_title_to_action_changes_when_shooting(), 'Shooting Move State', 1, 4,
                                     plots_path / ('specific_3_cat_move_not_move_shoot_2s' + str(trajectory_filter_options) + '.pdf'),
                                     fuse_not_changing_only=True)
    compute_move_not_move_histograms(get_title_to_action_changes_when_shooting(), 'Shooting Move State', 1, 4,
                                     plots_path / ('specific_2_cat_move_not_move_shoot_2s' + str(trajectory_filter_options) + '.pdf'),
                                     fuse_by_state_changing=True)

    compute_move_not_move_histograms(get_title_to_action_changes_when_killing(), 'Eliminating Move State', 1, 4,
                                     plots_path / ('specific_move_not_move_kill_2s' + str(trajectory_filter_options) + '.pdf'))
    compute_move_not_move_histograms(get_title_to_action_changes_when_killing(), 'Eliminating Move State', 1, 4,
                                     plots_path / ('specific_3_cat_move_not_move_kill_2s' + str(trajectory_filter_options) + '.pdf'),
                                     fuse_not_changing_only=True)
    compute_move_not_move_histograms(get_title_to_action_changes_when_killing(), 'Eliminating Move State', 1, 4,
                                     plots_path / ('specific_2_cat_move_not_move_kill_2s' + str(trajectory_filter_options) + '.pdf'),
                                     fuse_by_state_changing=True)
