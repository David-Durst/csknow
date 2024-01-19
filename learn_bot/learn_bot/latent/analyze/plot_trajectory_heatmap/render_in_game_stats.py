from math import ceil
from pathlib import Path
from pprint import pformat
from typing import List, Dict, Set, Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import plot_hist, generate_bins
from learn_bot.latent.analyze.plot_trajectory_heatmap.compute_teamwork_metrics import \
    get_title_to_places_to_round_counts, print_most_common_team_places, print_key_team_places, get_key_places_by_title, \
    get_all_places_by_title, num_players_col, ct_team_col, all_key_places, grouped_key_places, get_title_to_num_alive
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import get_title_to_speeds, \
    get_title_to_lifetimes, get_title_to_shots_per_kill
from learn_bot.libs.pd_printing import set_pd_print_options

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
        axs[ax_index, 0].set_xlabel(x_label)
        ax_index += 1
    plt.savefig(plot_file_path)


plt.rc('font', family='Arial')

title_rename_dict = {
    "1_15_24_learned_push": "CSMoveBot",
    "1_15_24_handcrafted": "ManualBot",
    "1_15_24_default": "CSGOBot",
}

def compute_one_metric_grid_histograms(title_to_values: Dict[str, List[float]], metric_title: str,
                                       bin_width: Union[int, float], max_bin_end: float, y_max: float,
                                       x_label: str, x_ticks: List, y_label: Optional[str],
                                       y_ticks: List, plot_file_path: Path):
    if len(title_to_values) != 4:
        return
    local_fig_length = 4
    fig = plt.figure(figsize=(local_fig_length * 2, local_fig_length * 2), constrained_layout=True)
    axs = fig.subplots(2, 2, squeeze=False, sharex=True, sharey=True)

    if y_label is not None:
        fig.suptitle(metric_title, x=0.52, fontsize=18)
        fig.supxlabel(x_label, x=0.52, fontsize=15)
        fig.supylabel(y_label, fontsize=15)
    else:
        fig.suptitle(metric_title, fontsize=18)
        fig.supxlabel(x_label, fontsize=15)

    if bin_width < 1.:
        num_bins = int(ceil(max_bin_end / bin_width))
        # add 1 as need left edge of every bin and right edge of last bin
        bins = [i * bin_width for i in range(num_bins + 1)]
    else:
        bins = generate_bins(0, int(ceil(max_bin_end)), bin_width)
    ax_index = 0
    for title, values in title_to_values.items():
        renamed_title = title
        if title in title_rename_dict:
            renamed_title = title_rename_dict[title]
        row_index = ax_index // 2
        col_index = ax_index % 2
        ax = axs[row_index, col_index]
        plot_hist(axs[row_index, col_index], pd.Series(values), bins)
        ax.set_xlim(0., max_bin_end)
        ax.set_ylim(0., y_max)
        ax.set_title(renamed_title, fontsize=15)
        #ax.set_xlabel(x_label)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)

        # remove right/top spine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # remove veritcal grid lines, make horizontal dotted
        ax.yaxis.grid(True, color='#EEEEEE', dashes=[4,1])
        ax.xaxis.grid(False)

        ax_index += 1
    plt.savefig(plot_file_path)

# want number of places with (1) 2 or 3 players and (2) by team - 4 rows
# first row is 2 players, CT
# second row is 2 players, T
# third row is 3 players, CT
# fourth row is 3 players, T
# columns are titles
def plot_most_common_places_by_title(plot_file_path: Path):
    key_places_by_title = get_key_places_by_title(all_key_places)
    places_by_title = get_all_places_by_title()

    titles = key_places_by_title.columns.tolist()
    num_options = 4
    fig = plt.figure(figsize=(fig_length * len(titles), fig_length * num_options), constrained_layout=True)
    axs = fig.subplots(num_options, len(titles), squeeze=False)
    num_players_options = [2, 2, 3, 3]
    ct_team_options = [True, False, True, False]
    for option_index in range(num_options):
        for title_index, title in enumerate(titles):
            num_players_option = num_players_options[option_index]
            ct_team_option = ct_team_options[option_index]
            sorted_places_by_title = places_by_title.sort_values(by=title, ascending=False)
            sorted_places_by_title = sorted_places_by_title[
                (sorted_places_by_title[num_players_col] == num_players_option) &
                (sorted_places_by_title[ct_team_col] == ct_team_option)]
            sorted_places_by_title = sorted_places_by_title.iloc[:4]
            sorted_places_by_title.loc[:, titles].plot(kind='bar', rot=45, title=f"{title} "
                                                                                 f"{num_players_option} Players "
                                                                                 f"{'CT' if ct_team_option else 'T'}",
                                                       ax=axs[option_index, title_index])
    plt.savefig(plot_file_path)


def plot_key_places(plot_path: Path):
    set_pd_print_options()
    title_to_num_alive = get_title_to_num_alive()
    for group, key_places in grouped_key_places.items():
        # plot key places
        key_places_by_title = get_key_places_by_title(key_places)
        key_places_by_title.plot(kind='bar', rot=90, title=group)
        plt.savefig(plot_path / (group.lower().replace(' ', '_') + '.png'), bbox_inches='tight')

        titles = key_places_by_title.columns.tolist()

        title_to_total_ticks = {}
        title_to_total_rounds = {}
        title_to_num_players_percent_of_ticks = {}
        title_to_num_players_total_rounds = {}
        title_to_num_players_at_start_total_rounds = {}
        title_to_num_players_rounds_in_group = {}
        title_to_num_players_percent_of_rounds = {}
        title_to_num_players_percent_of_start_rounds = {}
        for title in titles:
            # get overall data per title
            title_to_total_ticks[title] = title_to_num_alive[title].num_overall_ticks
            title_to_total_rounds[title] = title_to_num_alive[title].num_overall_rounds
            title_to_num_players_rounds_in_group[title] = key_places_by_title[title].sum()

            # get data that requires looking up by team in aggregate metrics
            if key_places[0].ct_team:
                title_to_num_players_percent_of_ticks[title] = \
                    title_to_num_alive[title].num_ct_alive_to_num_ticks[key_places[0].num_players()] / \
                    title_to_total_ticks[title]
                title_to_num_players_total_rounds[title] = \
                    title_to_num_alive[title].num_ct_alive_to_num_rounds[key_places[0].num_players()]
                title_to_num_players_at_start_total_rounds[title] = \
                    title_to_num_alive[title].num_ct_alive_at_start_to_num_rounds[key_places[0].num_players()]
            else:
                title_to_num_players_percent_of_ticks[title] = \
                    title_to_num_alive[title].num_t_alive_to_num_ticks[key_places[0].num_players()] / \
                    title_to_total_ticks[title]
                title_to_num_players_total_rounds[title] = \
                    title_to_num_alive[title].num_t_alive_to_num_rounds[key_places[0].num_players()]
                title_to_num_players_at_start_total_rounds[title] = \
                    title_to_num_alive[title].num_t_alive_at_start_to_num_rounds[key_places[0].num_players()]

            # this depends on data already looked up by team
            title_to_num_players_percent_of_rounds[title] = \
                title_to_num_players_rounds_in_group[title] / title_to_num_players_total_rounds[title]
            title_to_num_players_percent_of_start_rounds[title] = \
                title_to_num_players_rounds_in_group[title] / title_to_num_players_at_start_total_rounds[title]

        plot_name = group.lower().replace(' ', '_') + '_pct'
        with open(plot_path / (plot_name + '_num_rounds_ticks.txt'), 'w') as f:
            f.write('total ticks\n')
            f.write(pformat(title_to_total_ticks, indent=4))
            f.write('\ntotal rounds\n')
            f.write(pformat(title_to_total_rounds, indent=4))
            f.write('\ntotal num players percent of ticks\n')
            f.write(pformat(title_to_num_players_percent_of_ticks, indent=4))
            f.write('\nnum players total rounds\n')
            f.write(pformat(title_to_num_players_total_rounds, indent=4))
            f.write('\nnum players at start total rounds\n')
            f.write(pformat(title_to_num_players_at_start_total_rounds, indent=4))
            f.write('\ntotal num players rounds in group\n')
            f.write(pformat(title_to_num_players_rounds_in_group, indent=4))
            f.write('\npercent num players rounds in group\n')
            f.write(pformat(title_to_num_players_percent_of_rounds, indent=4))
            f.write('\npercent num players rounds at start in group\n')
            f.write(pformat(title_to_num_players_percent_of_start_rounds, indent=4))


        if len(titles) > 1:
            #title_to_percent_diff: Dict[str, pd.Series] = {}
            #for title in titles[1:]:
            #    title_to_percent_diff[title] = \
            #        ((key_places_by_title[titles[0]] - key_places_by_title[title]) / key_places_by_title[titles[0]]).abs()
            #title_to_percent_diff_df = pd.concat(title_to_percent_diff.values(), axis=1, keys=title_to_percent_diff.keys())
            #title_to_percent_diff_df.plot(kind='box')
            #plt.title(f"{group} Percent Diff to {titles[0]}")
            ##title_to_percent_mad_diff_series = pd.Series(title_to_percent_mad_diff)
            ##title_to_percent_mad_diff_series.plot(kind='bar')
            #plt.savefig(plot_path / (group.lower().replace(' ', '_') + '_pct.png'), bbox_inches='tight')

            title_to_percent_mad_diff: Dict[str, float] = {}
            title_to_abs_percent_diff: Dict[str, pd.Series] = {}
            for title in titles[1:]:
                title_to_percent_mad_diff[title] = \
                    ((key_places_by_title[titles[0]] - key_places_by_title[title]) / key_places_by_title[titles[0]]).abs().mean()
                title_to_abs_percent_diff[title] = \
                    ((key_places_by_title[titles[0]] - key_places_by_title[title]) / key_places_by_title[titles[0]]).abs()
            title_to_percent_diff_df = pd.concat(title_to_abs_percent_diff.values(), axis=1,
                                                 keys=title_to_abs_percent_diff.keys())
            fig, ax = plt.subplots()
            ax.bar(title_to_percent_mad_diff.keys(), title_to_percent_mad_diff.values())
            ax.set_ylabel('Percent MAD')
            ax.set_title(f"{group} {title} Percent MAD")
            plt.xticks(rotation=90)
            #title_to_percent_mad_diff_series = pd.Series(title_to_percent_mad_diff)
            #title_to_percent_mad_diff_series.plot(kind='bar')
            plt.savefig(plot_path / (plot_name + '.png'), bbox_inches='tight')
            with open(plot_path / (plot_name + '.txt'), 'w') as f:
                f.write(str(title_to_percent_diff_df.describe()))
            #print(group)
            #print(key_places_by_title)
            #print(pd.Series(title_to_percent_mad_diff))


def compute_metrics(trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    if trajectory_filter_options.is_no_filter():
        #plot_most_common_places_by_title(plots_path / 'most_popular_places.png')
        #key_places_by_title = get_key_places_by_title()
        #key_places_by_title.plot(kind='bar', rot=90, title='Rounds With Team Formations')
        #plt.savefig(plots_path / 'key_places.png', bbox_inches='tight')
        plot_key_places(plots_path)
    if trajectory_filter_options.compute_speeds:
        # airstrafing can get you above normal weapon max speed
        compute_one_metric_histograms(get_title_to_speeds(), 'Weapon/Scoped Scaled Speed', 0.1, 1., 'Percent Max Speed',
                                      plots_path / ('speeds_' + str(trajectory_filter_options) + '.png'))
    if trajectory_filter_options.compute_lifetimes:
        # small timing mismatch can get 41 seconds on bomb timer
        compute_one_metric_grid_histograms(get_title_to_lifetimes(), 'Lifetimes', 5, 40.,
                                           0.6, 'Lifetime Length (s)', [0, 10, 20, 30, 40], None, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                           plots_path / ('lifetimes_' + str(trajectory_filter_options) + '.pdf'))
    if trajectory_filter_options.compute_shots_per_kill:
        compute_one_metric_grid_histograms(get_title_to_shots_per_kill(), 'Shots Per Kill', 1, 30.,
                                           0.3, 'Shots', [0, 10, 20, 30], None, [0, 0.1, 0.2, 0.3],
                                           plots_path / ('shots_per_kill_' + str(trajectory_filter_options) + '.pdf'))
