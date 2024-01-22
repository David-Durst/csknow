from math import ceil
from pathlib import Path
from pprint import pformat
from typing import List, Dict, Set, Union, Optional

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import plot_hist, generate_bins
from learn_bot.latent.analyze.plot_trajectory_heatmap.title_rename_dict import title_rename_dict
from learn_bot.latent.analyze.plot_trajectory_heatmap.compute_teamwork_metrics import \
    get_title_to_places_to_round_counts, print_most_common_team_places, print_key_team_places, get_key_places_by_title, \
    get_all_places_by_title, num_players_col, ct_team_col, all_key_places, grouped_key_places, get_title_to_num_alive, \
    get_title_to_opportunities_for_a_site_mistake, get_title_to_num_a_site_mistakes, get_title_to_num_b_site_mistakes, \
    get_title_to_opportunities_for_b_site_mistake, get_title_to_opportunities_for_a_site_round_mistake, \
    get_title_to_num_a_site_round_mistakes, get_title_to_num_b_site_round_mistakes, \
    get_title_to_opportunities_for_b_site_round_mistake
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


def compute_one_metric_four_histograms(title_to_values: Dict[str, List[float]], metric_title: str,
                                       bin_width: Union[int, float], max_bin_end: float, y_max: float,
                                       x_label: str, x_ticks: List, y_label: Optional[str],
                                       y_ticks: List, plot_file_path: Path):
    if len(title_to_values) != 4:
        return
    local_fig_length = 3.3
    fig = plt.figure(figsize=(local_fig_length, 0.5 + local_fig_length / 3), constrained_layout=True)
    axs = fig.subplots(1, 4, squeeze=False, sharey=True)

    if y_label is not None:
        fig.suptitle(metric_title, x=0.52, fontsize=8)
        fig.supxlabel(x_label, x=0.52, fontsize=8)
        fig.supylabel(y_label, fontsize=8)
    else:
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
        ax.yaxis.grid(True, color='#EEEEEE', dashes=[4,1])
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



# want number of places with (1) 2 or 3 players and (2) by team - 4 rows
# first row is 2 players, CT
# second row is 2 players, T
# third row is 3 players, CT
# fourth row is 3 players, T
# columns are titles
def plot_most_common_places_by_title(plot_file_path: Path):
    key_places_by_title = get_key_places_by_title(all_key_places, False)
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


name_to_ylim = {
    "Offense Flanks": 100,
    "Defense Spread": 100,
    "mistakes": 50
}

name_to_yticks = {
    "Offense Flanks": [0, 20, 40, 60, 80, 100],
    "Defense Spread": [0, 20, 40, 60, 80, 100],
    "mistakes": [0, 10, 20, 30, 40, 50]
}

situation_rename_dict = {
    "T A ExtendedA,BombsiteA,LongA": "D1",
    "T A BombsiteA,BombsiteA,LongA": "D2",
    "T A BombsiteA,LongA,ARamp": "D3",
    "T B BombsiteB,UpperTunnel,BDoors": "D4",
    "T B BombsiteB,BombsiteB,BDoors": "D5",
    "T B BombsiteB,BombsiteB,UpperTunnel": "D6",
    "CT A LongA,ShortStairs": "A1",
    "CT A CTSpawn,ShortStairs": "A2",
    "CT A CTSpawn,LongA": "A3",
    "CT B UpperTunnel,BDoors": "A4",
    "CT B UpperTunnel,Hole": "A5",
}


def plot_place_title_df(df: pd.DataFrame, chart_name: str, plot_file_path: Path, y_label: str):
    df.index = df.index.to_series().replace(situation_rename_dict)
    df.rename(title_rename_dict, axis=1, inplace=True)

    fig, ax = plt.subplots()

    df.plot(kind='bar', title=chart_name, rot=0, ax=ax)#, color="#3f8f35")
    if chart_name in name_to_ylim:
        #ax.set_ylim(0., name_to_ylim[chart_name])
        #ax.set_yticks(name_to_yticks[chart_name])
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
    ax.set_title(chart_name, fontsize=15)
    # ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    #ax.set_xticks(x_ticks)

    # remove right/top spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    ax.yaxis.grid(True, color='#EEEEEE', dashes=[4, 1])
    ax.xaxis.grid(False)
    plt.savefig(plot_file_path, bbox_inches='tight')


def plot_key_places(plot_path: Path, use_tick_counts: bool):
    set_pd_print_options()
    title_to_num_alive = get_title_to_num_alive()
    for group, key_places in grouped_key_places.items():
        # plot key places
        key_places_by_title = get_key_places_by_title(key_places, use_tick_counts)
        key_places_by_title_copy = key_places_by_title.copy()
        if use_tick_counts:
            key_places_by_title_copy /= 16
        if use_tick_counts:
            plots_file_name = plot_path / (group.lower().replace(' ', '_') + '_ticks.pdf')
        else:
            plots_file_name = plot_path / (group.lower().replace(' ', '_') + '_rounds.pdf')
        plot_place_title_df(key_places_by_title_copy, group, plots_file_name,
                            'Seconds' if use_tick_counts else 'Rounds')

        if use_tick_counts:
            continue

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
                describe_df = title_to_percent_diff_df.describe()
                f.write(str(describe_df))
                f.write('\n')
                f.write('IQR: ')
                f.write(str(describe_df.loc['75%'] - describe_df.loc['25%']))
            #print(group)
            #print(key_places_by_title)
            #print(pd.Series(title_to_percent_mad_diff))


def plot_mistakes(plots_path, use_tick_counts: bool):
    if use_tick_counts:
        mistakes_df = pd.DataFrame.from_records([get_title_to_num_a_site_round_mistakes(),
                                                 get_title_to_num_b_site_round_mistakes()],
                                                index=['BombsiteA', 'BombsiteB'])
    else:
        mistakes_df = pd.DataFrame.from_records([get_title_to_num_a_site_mistakes(),
                                                 get_title_to_num_b_site_mistakes()],
                                                index=['BombsiteA', 'BombsiteB'])

    title = 'Mistakes' if len([s for s in mistakes_df.columns if 'default' in s]) > 0 else 'Ablation Mistakes'
    plot_place_title_df(mistakes_df, title, plots_path / 'mistakes.pdf', 'Events' if use_tick_counts else 'Rounds')

    if not use_tick_counts:
        with open(plots_path / 'mistakes.txt', 'w') as f:
            f.write('a opportunities\n')
            f.write(pformat(get_title_to_opportunities_for_a_site_mistake(), indent=4))
            f.write('a round opportunities\n')
            f.write(pformat(get_title_to_opportunities_for_a_site_round_mistake(), indent=4))
            f.write('a mistakes\n')
            f.write(pformat(get_title_to_num_a_site_mistakes(), indent=4))
            f.write('a round mistakes\n')
            f.write(pformat(get_title_to_num_a_site_round_mistakes(), indent=4))

            f.write('b opportunities\n')
            f.write(pformat(get_title_to_opportunities_for_b_site_mistake(), indent=4))
            f.write('b round opportunities\n')
            f.write(pformat(get_title_to_opportunities_for_b_site_round_mistake(), indent=4))
            f.write('b mistakes\n')
            f.write(pformat(get_title_to_num_b_site_mistakes(), indent=4))
            f.write('b round mistakes\n')
            f.write(pformat(get_title_to_num_b_site_round_mistakes(), indent=4))


def compute_metrics(trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    if trajectory_filter_options.is_no_filter():
        #plot_most_common_places_by_title(plots_path / 'most_popular_places.png')
        #key_places_by_title = get_key_places_by_title()
        #key_places_by_title.plot(kind='bar', rot=90, title='Rounds With Team Formations')
        #plt.savefig(plots_path / 'key_places.png', bbox_inches='tight')
        plot_key_places(plots_path, True)
        plot_key_places(plots_path, False)
        plot_mistakes(plots_path, True)
        plot_mistakes(plots_path, False)



    if trajectory_filter_options.compute_speeds:
        # airstrafing can get you above normal weapon max speed
        compute_one_metric_histograms(get_title_to_speeds(), 'Weapon/Scoped Scaled Speed', 0.1, 1., 'Percent Max Speed',
                                      plots_path / ('speeds_' + str(trajectory_filter_options) + '.png'))
    if trajectory_filter_options.compute_lifetimes:
        # small timing mismatch can get 41 seconds on bomb timer
        compute_one_metric_four_histograms(get_title_to_lifetimes(), 'Lifetimes', 5, 40.,
                                           0.6, 'Seconds', [0, 20, 40], None, [0, 0.3, 0.6],
                                           plots_path / ('lifetimes_' + str(trajectory_filter_options) + '.pdf'))
        compute_one_metric_emd(get_title_to_lifetimes(),
                               plots_path / ('lifetimes_' + str(trajectory_filter_options) + '.txt'))
    if trajectory_filter_options.compute_shots_per_kill:
        compute_one_metric_four_histograms(get_title_to_shots_per_kill(), 'Shots Per Kill', 1, 30.,
                                           0.3, 'Shots', [0, 15, 30], None, [0, 0.15, 0.3],
                                           plots_path / ('shots_per_kill_' + str(trajectory_filter_options) + '.pdf'))
        compute_one_metric_emd(get_title_to_shots_per_kill(),
                               plots_path / ('shots_per_kill_' + str(trajectory_filter_options) + '.txt'))
