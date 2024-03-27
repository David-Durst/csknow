from pathlib import Path
from pprint import pformat
from typing import List, Dict, Optional

import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.plot_trajectory_heatmap.compute_teamwork_metrics import get_key_places_by_title, \
    all_key_places, get_all_places_by_title, num_players_col, ct_team_col, grouped_key_places, \
    offense_two_man_flanks_str, defense_spread_str, get_title_to_num_alive, get_title_to_num_a_site_round_mistakes, \
    get_title_to_num_b_site_round_mistakes, get_title_to_num_a_site_mistakes, get_title_to_num_b_site_mistakes, \
    get_title_to_opportunities_for_a_site_mistake, get_title_to_opportunities_for_a_site_round_mistake, \
    get_title_to_opportunities_for_b_site_mistake, get_title_to_opportunities_for_b_site_round_mistake
from learn_bot.latent.analyze.plot_trajectory_heatmap.title_rename_dict import title_rename_dict
from learn_bot.libs.pd_printing import set_pd_print_options


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
    fig = plt.figure(figsize=(6 * len(titles), 6 * num_options), constrained_layout=True)
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
    "T A ExtendedA,BombsiteA,LongA": "S1",
    "T A BombsiteA,BombsiteA,LongA": "S2",
    "T A BombsiteA,LongA,ARamp": "S3",
    "T B BombsiteB,UpperTunnel,BDoors": "S4",
    "T B BombsiteB,BombsiteB,BDoors": "S5",
    "T B BombsiteB,BombsiteB,UpperTunnel": "S6",
    "CT A LongA,ShortStairs": "F1",
    "CT A CTSpawn,ShortStairs": "F2",
    "CT A CTSpawn,LongA": "F3",
    "CT B UpperTunnel,BDoors": "F4",
    "CT B UpperTunnel,Hole": "F5",
}
legend_pos_dict = {
    "Mistakes": (0.375, 0.4)
}
no_legend_set = {'Defense Spread'}


def plot_place_title_df(df: pd.DataFrame, chart_name: str, plot_file_path: Path, y_label: str, y_ticks: List,
                        margin_df: Optional[pd.DataFrame]):
    df.index = df.index.to_series().replace(situation_rename_dict)
    df.rename(title_rename_dict, axis=1, inplace=True)

    fig, ax = plt.subplots(figsize=(3.3, 3.3*0.6))

    df.plot(kind='bar', title=chart_name, rot=0, ax=ax)#, color=default_bar_colro)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(chart_name, fontsize=8)
    # ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_yticks(y_ticks)

    #ax.set_xticks(x_ticks)

    # remove right/top spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    ax.yaxis.grid(True, color='#EEEEEE', dashes=[4, 1])
    ax.xaxis.grid(False)
    if chart_name in legend_pos_dict:
        plt.legend(bbox_to_anchor=legend_pos_dict[chart_name], bbox_transform=ax.transAxes, fontsize=8)
    else:
        plt.legend(fontsize=8)
    plt.savefig(plot_file_path, bbox_inches='tight')


def plot_specific_key_places(plot_path: Path):
    fig, axs = plt.subplots(figsize=(3.3, 3.3*0.6 * 2), nrows=2, ncols=1)

    offense_key_places = get_key_places_by_title(grouped_key_places[offense_two_man_flanks_str], False)
    offense_key_places.index = offense_key_places.index.to_series().replace(situation_rename_dict)
    offense_key_places.rename(title_rename_dict, axis=1, inplace=True)

    offense_key_places.plot(kind='bar', rot=0, ax=axs[0])#, color=default_bar_color)
    axs[0].tick_params(axis="x", labelsize=8)
    axs[0].tick_params(axis="y", labelsize=8)
    axs[0].set_title("Offense Flank Occurrences", fontsize=8)
    # ax.set_xlabel(x_label)
    axs[0].set_ylabel('Rounds', fontsize=8)
    axs[0].set_yticks([0, 40, 80])

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    axs[0].yaxis.grid(True, color='#EEEEEE', dashes=[4, 1])
    axs[0].xaxis.grid(False)
    #plt.legend(bbox_to_anchor=legend_pos_dict[offense_two_man_flanks_str], bbox_transform=axs[0].transAxes, fontsize=8)
    #plt.legend(fontsize=8)
    axs[0].get_legend().remove()


    defense_key_places = get_key_places_by_title(grouped_key_places[defense_spread_str], False)
    defense_key_places.index = defense_key_places.index.to_series().replace(situation_rename_dict)
    defense_key_places.rename(title_rename_dict, axis=1, inplace=True)

    defense_key_places.plot(kind='bar', rot=0, ax=axs[1])#, color=default_bar_color)
    axs[1].tick_params(axis="x", labelsize=8)
    axs[1].tick_params(axis="y", labelsize=8)
    axs[1].set_title("Defense Spread Occurrences", fontsize=8)
    # ax.set_xlabel(x_label)
    axs[1].set_ylabel('Rounds', fontsize=8)
    axs[1].set_yticks([0, 40, 80])

    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    axs[1].yaxis.grid(True, color='#EEEEEE', dashes=[4, 1])
    axs[1].xaxis.grid(False)
    #axs[1].legend(bbox_to_anchor=(2.5, -0.15), loc='center', bbox_transform=axs[1].transAxes, fontsize=8, ncol=len(defense_key_places.columns))
    axs[1].legend(bbox_to_anchor=(0.45, -0.15), loc='upper center', fontsize=8, ncol=len(defense_key_places.columns) // 2)
    #axs[1].legend(loc=())
    #plt.subplots_adjust(left=0.065, right=0.97, top=0.96, bottom=0.065, wspace=0.14)
    #fig.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.95, hspace=0.35)
    plt.savefig(plot_path / 'specific_key_places_rounds.pdf', bbox_inches='tight')


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
                            'Seconds' if use_tick_counts else 'Rounds', [0, 40, 80])

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
                f.write('\n')
                f.write('values: ')
                f.write(str(title_to_percent_diff_df))
            key_places_by_title.to_csv(plot_path / (plot_name + '_for_aggregation.csv'))
            #print(group)
            #print(key_places_by_title)
            #print(pd.Series(title_to_percent_mad_diff))


def plot_mistakes(plots_path, use_tick_counts: bool):
    if use_tick_counts:
        mistakes_df = pd.DataFrame.from_records([get_title_to_num_a_site_round_mistakes(),
                                                 get_title_to_num_b_site_round_mistakes()],
                                                index=['Leave \n High Ground', 'Leave \n Established Position'])
    else:
        mistakes_df = pd.DataFrame.from_records([get_title_to_num_a_site_mistakes(),
                                                 get_title_to_num_b_site_mistakes()],
                                                index=['Leave \n High Ground', 'Leave \n Established Position'])

    if len([s for s in mistakes_df.columns if 'default' in s]) > 0:
        title = 'Mistakes'
        y_ticks = [0, 100, 200]
    else:
        title = 'Ablation Mistakes'
        y_ticks = [0, 20, 40]
    mistakes_name = 'mistakes_ticks.pdf' if use_tick_counts else 'mistakes.pdf'
    plot_place_title_df(mistakes_df, title, plots_path / mistakes_name, 'Events' if use_tick_counts else 'Rounds',
                        y_ticks)

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
        mistakes_df.to_csv(plots_path / 'mistakes_aggregation.csv')
