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


def compute_one_grouped_metric_histograms(title_to_groups_to_values: Dict[str, Dict[int, List[float]]], metric_title: str,
                                          group_title: str, bin_width: Union[int, float], smallest_max: float,
                                          x_label: str, plot_file_path: Path):
    num_titles = len(title_to_groups_to_values.keys())
    unique_groups: Set[int] = set()
    for _, group_to_values in title_to_groups_to_values.items():
        for group in group_to_values.keys():
            unique_groups.add(group)
    fig = plt.figure(figsize=(fig_length * len(unique_groups), fig_length * num_titles), constrained_layout=True)
    axs = fig.subplots(num_titles, len(unique_groups), squeeze=False)

    # get the max value for histogram (if bigger than smaller max), min wil be 0 always
    max_observed = smallest_max
    for _, group_to_values in title_to_groups_to_values.items():
        for _, values in group_to_values.items():
            max_observed = max(max_observed, max(values))

    if bin_width < 1.:
        num_bins = int(ceil(max_observed / bin_width))
        # add 1 as need left edge of every bin and right edge of last bin
        bins = [i * bin_width for i in range(num_bins + 1)]
    else:
        bins = generate_bins(0, int(ceil(max_observed)), bin_width)
    for title_index, (title, group_to_values) in enumerate(title_to_groups_to_values.items()):
        for group_index, group in enumerate(unique_groups):
            if group not in group_to_values:
                continue
            plot_hist(axs[title_index, group_index], pd.Series(group_to_values[group]), bins)
            axs[title_index, group_index].set_xlim(0., max_observed)
            axs[title_index, group_index].set_ylim(0., 1.)
            axs[title_index, group_index].set_title(title + " " + metric_title + " " + group_title + " " + str(group))
            axs[title_index, group_index].set_xlabel(x_label)
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
        title_to_num_players_rounds_in_group = {}
        title_to_num_players_percent_of_rounds = {}
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
            else:
                title_to_num_players_percent_of_ticks[title] = \
                    title_to_num_alive[title].num_t_alive_to_num_ticks[key_places[0].num_players()] / \
                    title_to_total_ticks[title]
                title_to_num_players_total_rounds[title] = \
                    title_to_num_alive[title].num_t_alive_to_num_rounds[key_places[0].num_players()]

            # this depends on data already looked up by team
            title_to_num_players_percent_of_rounds[title] = \
                title_to_num_players_rounds_in_group[title] / title_to_num_players_total_rounds[title]

        plot_name = group.lower().replace(' ', '_') + '_pct'
        with open(plot_path / (plot_name + '_num_rounds_ticks.txt'), 'w') as f:
            f.write('total ticks\n')
            f.write(pformat(title_to_total_ticks, indent=4))
            f.write('total rounds\n')
            f.write(pformat(title_to_total_rounds, indent=4))
            f.write('total num players percent of ticks\n')
            f.write(pformat(title_to_num_players_percent_of_ticks, indent=4))
            f.write('total num players total rounds\n')
            f.write(pformat(title_to_num_players_total_rounds, indent=4))
            f.write('total num players rounds in group\n')
            f.write(pformat(title_to_num_players_rounds_in_group, indent=4))
            f.write('total num players percent rounds in group\n')
            f.write(pformat(title_to_num_players_percent_of_rounds, indent=4))


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
            print(group)
            #print(key_places_by_title)
            print(pd.Series(title_to_percent_mad_diff))


def compute_metrics(trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    if trajectory_filter_options.is_no_filter():
        #plot_most_common_places_by_title(plots_path / 'most_popular_places.png')
        #key_places_by_title = get_key_places_by_title()
        #key_places_by_title.plot(kind='bar', rot=90, title='Rounds With Team Formations')
        #plt.savefig(plots_path / 'key_places.png', bbox_inches='tight')
        plot_key_places(plots_path)
    #if trajectory_filter_options.is_no_filter():
        #compute_one_grouped_metric_histograms(get_title_to_num_teammates_to_enemy_vis_on_death(), 'Teammate Saw Enemy On Death',
        #                                      'Teammates', 0.1, 1., 'Min Time of Alive Teammates (s)',
        #                                      plots_path / ('teammate_vis_' + str(trajectory_filter_options) + '.png'))
        #compute_one_grouped_metric_histograms(get_title_to_num_enemies_to_my_team_vis_on_death(),
        #                                      'Enemy Saw My Team On Death',
        #                                      'Enemies', 0.1, 1., 'Min Time of Alive Enemies (s)',
        #                                      plots_path / ('enemy_vis_' + str(trajectory_filter_options) + '.png'))
        #compute_one_grouped_metric_histograms(get_title_to_num_teammates_to_distance_to_teammate_on_death(),
        #                                      'Teammate Distances On Death',
        #                                      'Teammates', 100, 1000, 'Teammate Distances (Hammer Units)',
        #                                      plots_path / ('teammate_distances_' + str(trajectory_filter_options) + '.png'))
        #compute_one_grouped_metric_histograms(get_title_to_num_enemies_to_distance_to_enemy_on_death(),
        #                                      'Enemy Distances On Death',
        #                                      'Enemies', 100, 1000, 'Enemy Distances (Hammer Units)',
        #                                      plots_path / ('enemy_distances_' + str(trajectory_filter_options) + '.png'))
        #compute_one_grouped_metric_histograms(get_title_to_num_teammates_to_distance_multi_engagements(),
        #                                      'Distance On Multi Engagements',
        #                                      'Teammates', 100, 1000, 'Teammate Distances (Hammer Units)',
        #                                      plots_path / ('distances_multi_engagements_' + str(trajectory_filter_options) + '.png'))
        #compute_one_metric_histograms(get_title_to_blocking_events(), 'Blocking Events', 1, 30., 'Blocking Events Per Round',
        #                              plots_path / ('blocking_' + str(trajectory_filter_options) + '.png'))
        #compute_one_metric_histograms(get_title_to_num_multi_engagements(), 'Num Multi-Engagements', 1, 30.,
        #                              'Num Multi-Engagements Per Round',
        #                              plots_path / ('num_multi_engagements_' + str(trajectory_filter_options) + '.png'))
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
