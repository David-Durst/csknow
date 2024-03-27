import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from einops import rearrange
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import iqr

from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path
from learn_bot.latent.analyze.plot_trajectory_heatmap.render_key_places_and_mistakes import plot_place_title_df, \
    situation_rename_dict
from learn_bot.latent.analyze.plot_trajectory_heatmap.title_rename_dict import title_rename_dict


@dataclass
class EventAggregation:
    per_event_median_df: pd.DataFrame
    per_event_iqr_df: pd.DataFrame
    all_events_median_iqr_df: pd.DataFrame

    def to_csv(self, plots_path: Path, event_name: str):
        self.per_event_median_df.to_csv(plots_path / (event_name + "_per_event_median.csv"))
        self.per_event_iqr_df.to_csv(plots_path / (event_name + "_per_event_iqr.csv"))
        self.all_events_median_iqr_df.to_csv(plots_path / (event_name + "_all_events.csv"))


def aggregate_one_event_type(rollout_extensions: list[str], rollout_prefix: str, event_csv_path: Path,
                             diff_to_first_player_type: bool, aggregation_plots_path: Path, event_type: str) -> EventAggregation:
    event_nps: List[np.ndarray] = []

    for i, rollout_extension in enumerate(rollout_extensions):
        plots_path = similarity_plots_path / (rollout_prefix + rollout_extension)
        new_event_df = pd.read_csv(plots_path / event_csv_path, index_col=0)
        if i == 0:
            column_names = new_event_df.columns
            row_index = new_event_df.index
        event_nps.append(new_event_df.values)

    event_np = np.stack(event_nps, axis=-1)

    per_event_median_np = np.median(event_np, axis=2)
    per_event_median_df = pd.DataFrame(data=per_event_median_np, index=row_index, columns=column_names)
    per_event_iqr_np = iqr(event_np, axis=2)
    per_event_iqr_df = pd.DataFrame(data=per_event_iqr_np, index=row_index, columns=column_names)

    if diff_to_first_player_type:
        column_names = column_names[1:]
        event_np = np.abs((event_np[:, 1:, :] - event_np[:, [0], :]) / event_np[:, [0], :])

    # in order to aggregate across different events, get player type first dimension,
    # then event type, then different trials of events
    player_event_trials_np = rearrange(event_np, 'i j k -> j (i k)')
    all_events_median_np = np.median(player_event_trials_np, axis=1)
    all_events_iqr_np = iqr(player_event_trials_np, axis=1)
    all_events_median_iqr_df = \
        pd.DataFrame(data=[all_events_median_np, all_events_iqr_np], columns=column_names, index=['Median', 'IQR'])
    result = EventAggregation(per_event_median_df, per_event_iqr_df, all_events_median_iqr_df)
    result.to_csv(aggregation_plots_path, event_type)
    return result


def plot_offense_defense(offense_events: EventAggregation, defense_events: EventAggregation,
                         aggregation_plots_path: Path):
    fig, axs = plt.subplots(figsize=(3.3, 3.3*0.6 * 2), nrows=2, ncols=1)

    offense_events_median_df = offense_events.per_event_median_df
    offense_events_median_df.index = offense_events_median_df.index.to_series().replace(situation_rename_dict)
    offense_events_median_df.rename(title_rename_dict, axis=1, inplace=True)
    offense_events_iqr_df = offense_events.per_event_iqr_df
    offense_events_iqr_df.index = offense_events_iqr_df.index.to_series().replace(situation_rename_dict)
    offense_events_iqr_df.rename(title_rename_dict, axis=1, inplace=True)

    offense_events_median_df.plot(kind='bar', rot=0, ax=axs[0], yerr=offense_events_iqr_df)#, color=default_bar_color)
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


    defense_events_median_df = defense_events.per_event_median_df
    defense_events_median_df.index = defense_events_median_df.index.to_series().replace(situation_rename_dict)
    defense_events_median_df.rename(title_rename_dict, axis=1, inplace=True)
    defense_events_iqr_df = defense_events.per_event_iqr_df
    defense_events_iqr_df.index = defense_events_iqr_df.index.to_series().replace(situation_rename_dict)
    defense_events_iqr_df.rename(title_rename_dict, axis=1, inplace=True)

    defense_events_median_df.plot(kind='bar', rot=0, ax=axs[1], yerr=defense_events_iqr_df)#, color=default_bar_color)
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
    axs[1].legend(bbox_to_anchor=(0.45, -0.15), loc='upper center', fontsize=8, ncol=len(defense_events.per_event_median_df.columns) // 2)
    #axs[1].legend(loc=())
    #plt.subplots_adjust(left=0.065, right=0.97, top=0.96, bottom=0.065, wspace=0.14)
    #fig.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.95, hspace=0.35)
    plt.savefig(aggregation_plots_path / 'specific_key_places_rounds.pdf', bbox_inches='tight')


def plot_mistakes(mistakes_events: EventAggregation, aggregation_plots_path: Path):
    y_ticks = [0, 100, 200]
    if mistakes_events.per_event_median_df.max().max() < 80:
        y_ticks = [0, 20, 40]
    plot_place_title_df(mistakes_events.per_event_median_df, 'Mistakes', aggregation_plots_path / 'mistakes.pdf',
                        'Rounds', y_ticks, mistakes_events.per_event_iqr_df)



def aggregate_trajectory_events(rollout_extensions: list[str], rollout_prefix: str):
    aggregation_plots_path = similarity_plots_path / ("agg_" + rollout_prefix + rollout_extensions[0])
    aggregation_plots_path.mkdir(parents=True, exist_ok=True)

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    offense_events = aggregate_one_event_type(rollout_extensions, rollout_prefix,
                                              Path("offense_flanks_pct_for_aggregation.csv"), True,
                                              aggregation_plots_path, 'offense_flanks')
    defense_events = aggregate_one_event_type(rollout_extensions, rollout_prefix,
                                              Path("defense_spread_pct_for_aggregation.csv"), True,
                                              aggregation_plots_path, 'defense_spread')
    plot_offense_defense(offense_events, defense_events, aggregation_plots_path)

    mistakes_events = aggregate_one_event_type(rollout_extensions, rollout_prefix,
                                               Path("mistakes_aggregation.csv"), True,
                                               aggregation_plots_path, 'mistakes')
    plot_mistakes(mistakes_events, aggregation_plots_path)
    return

    aggregate_one_event_type(rollout_extensions, rollout_prefix, Path("diff") / "emd_no_filter.txt", False,
                             aggregation_plots_path, 'emd_no_filter')
    aggregate_one_event_type(rollout_extensions, rollout_prefix, Path("diff") / "emd_only_kill.txt", False,
                             aggregation_plots_path, 'emd_only_kill')

if __name__ == "__main__":
    rollout_extensions = sys.argv[1].split(',')
    rollout_prefix = sys.argv[2] if len(sys.argv) >= 3 else ""
    aggregate_trajectory_events(rollout_extensions, rollout_prefix)