import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from cycler import cycler
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
class MetricAggregation:
    per_event_median_df: pd.DataFrame
    per_event_iqr_df: pd.DataFrame
    all_events_median_iqr_df: pd.DataFrame
    per_event_delta_median_df: Optional[pd.DataFrame]
    per_event_delta_iqr_df: Optional[pd.DataFrame]

    def to_csv(self, plots_path: Path, event_name: str):
        self.per_event_median_df.to_csv(plots_path / (event_name + "_per_event_median.csv"))
        self.per_event_iqr_df.to_csv(plots_path / (event_name + "_per_event_iqr.csv"))
        self.all_events_median_iqr_df.to_csv(plots_path / (event_name + "_all_events.csv"))
        if self.per_event_delta_median_df is not None:
            self.per_event_delta_median_df.to_csv(plots_path / (event_name + "_per_event_delta_median.csv"))
            self.per_event_delta_iqr_df.to_csv(plots_path / (event_name + "_per_event_delta_iqr.csv"))


def aggregate_one_metric_type(rollout_extensions: list[str], rollout_prefix: str, metric_data_path: Path,
                              diff_to_first_player_type: bool, aggregation_plots_path: Path, event_type: str,
                              # set if only one event in the event (like lifetimes), so need to convert series to dataframe
                              event_name: Optional[str] = None,
                              # set if events need to be sorted (like lifetimes and shots per kill)
                              sorted_example_path: Optional[Path] = None) -> MetricAggregation:
    event_nps: List[np.ndarray] = []

    for i, rollout_extension in enumerate(rollout_extensions):
        plots_path = similarity_plots_path / (rollout_prefix + rollout_extension)
        if event_name is not None:
            with open(plots_path / metric_data_path, 'r') as f:
                new_event_dict = eval(f.read())
            # make lists so pandas accepts as one row df
            new_event_dict = {k: [v] for k, v in new_event_dict.items()}
            new_event_df = pd.DataFrame.from_dict(new_event_dict)
            new_event_df.index = [event_name]
            if sorted_example_path:
                sorted_example_df = pd.read_csv(plots_path / sorted_example_path, index_col=0)
                new_event_df = new_event_df[sorted_example_df.index]
        else:
            new_event_df = pd.read_csv(plots_path / metric_data_path, index_col=0)
        if i == 0:
            column_names = new_event_df.columns
            row_index = new_event_df.index
        event_nps.append(new_event_df.values)

    event_np = np.stack(event_nps, axis=-1)

    per_event_median_np = np.median(event_np, axis=2)
    per_event_median_df = pd.DataFrame(data=per_event_median_np, index=row_index, columns=column_names)
    per_event_iqr_np = iqr(event_np, axis=2)
    per_event_iqr_df = pd.DataFrame(data=per_event_iqr_np, index=row_index, columns=column_names)

    per_event_delta_median_df = None
    per_event_delta_iqr_df = None
    if diff_to_first_player_type:
        column_names = column_names[1:]
        event_delta_np = np.abs(event_np[:, 1:, :] - event_np[:, [0], :])
        event_np = np.abs((event_np[:, 1:, :] - event_np[:, [0], :]) / event_np[:, [0], :])
        per_event_delta_median_np = np.median(event_delta_np, axis=2)
        per_event_delta_median_df = pd.DataFrame(data=per_event_delta_median_np, index=row_index, columns=column_names)
        per_event_delta_iqr_np = iqr(event_delta_np, axis=2)
        per_event_delta_iqr_df = pd.DataFrame(data=per_event_delta_iqr_np, index=row_index, columns=column_names)

    # in order to aggregate across different events, get player type first dimension,
    # then event type, then different trials of events
    player_event_trials_np = rearrange(event_np, 'i j k -> j (i k)')
    all_events_median_np = np.median(player_event_trials_np, axis=1)
    all_events_iqr_np = iqr(player_event_trials_np, axis=1)
    all_events_median_iqr_df = \
        pd.DataFrame(data=[all_events_median_np, all_events_iqr_np], columns=column_names, index=['Median', 'IQR'])
    result = MetricAggregation(per_event_median_df, per_event_iqr_df, all_events_median_iqr_df,
                               per_event_delta_median_df, per_event_delta_iqr_df)
    result.to_csv(aggregation_plots_path, event_type)
    return result


def plot_offense_defense(offense_events: MetricAggregation, defense_events: MetricAggregation,
                         aggregation_plots_path: Path):
    # add extra height for label
    fig, axs = plt.subplots(figsize=(3.3, 3.3*0.6 * 2 * 1.15), nrows=2, ncols=1)

    #next(axs[0]._get_lines.prop_cycler)
    #next(axs[1]._get_lines.prop_cycler)
    offense_events_median_df = offense_events.per_event_delta_median_df
    offense_events_median_df.index = offense_events_median_df.index.to_series().replace(situation_rename_dict)
    offense_events_median_df.rename(title_rename_dict, axis=1, inplace=True)
    offense_events_iqr_df = offense_events.per_event_delta_iqr_df
    offense_events_iqr_df.index = offense_events_iqr_df.index.to_series().replace(situation_rename_dict)
    offense_events_iqr_df.rename(title_rename_dict, axis=1, inplace=True)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
    offense_events_median_df.plot(kind='bar', rot=0, ax=axs[0], color=colors, yerr=offense_events_iqr_df)#, color=default_bar_color)
    axs[0].tick_params(axis="x", labelsize=8)
    axs[0].tick_params(axis="y", labelsize=8)
    axs[0].set_title("Offense Flank Occurrence Errors", fontsize=8)
    # ax.set_xlabel(x_label)
    axs[0].set_ylabel('Rounds', fontsize=8)
    axs[0].set_yticks([0, 30, 60])
    axs[0].set_ylim(bottom=0)

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    axs[0].yaxis.grid(True, color='#EEEEEE', dashes=[4, 1])
    axs[0].xaxis.grid(False)
    #plt.legend(bbox_to_anchor=legend_pos_dict[offense_two_man_flanks_str], bbox_transform=axs[0].transAxes, fontsize=8)
    #plt.legend(fontsize=8)
    axs[0].get_legend().remove()


    defense_events_median_df = defense_events.per_event_delta_median_df
    defense_events_median_df.index = defense_events_median_df.index.to_series().replace(situation_rename_dict)
    defense_events_median_df.rename(title_rename_dict, axis=1, inplace=True)
    defense_events_iqr_df = defense_events.per_event_delta_iqr_df
    defense_events_iqr_df.index = defense_events_iqr_df.index.to_series().replace(situation_rename_dict)
    defense_events_iqr_df.rename(title_rename_dict, axis=1, inplace=True)

    defense_events_median_df.plot(kind='bar', rot=0, ax=axs[1], color=colors, yerr=defense_events_iqr_df)#, color=default_bar_color)
    axs[1].tick_params(axis="x", labelsize=8)
    axs[1].tick_params(axis="y", labelsize=8)
    axs[1].set_title("Defense Spread Occurrence Errors", fontsize=8)
    # ax.set_xlabel(x_label)
    axs[1].set_ylabel('Rounds', fontsize=8)
    axs[1].set_yticks([0, 30, 60])
    axs[1].set_ylim(bottom=0)

    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    axs[1].yaxis.grid(True, color='#EEEEEE', dashes=[4, 1])
    axs[1].xaxis.grid(False)
    #axs[1].legend(bbox_to_anchor=(2.5, -0.15), loc='center', bbox_transform=axs[1].transAxes, fontsize=8, ncol=len(defense_key_places.columns))
    axs[1].legend(bbox_to_anchor=(0.45, -0.15), loc='upper center', fontsize=8, ncol=len(defense_events.per_event_median_df.columns) // 1)
    #axs[1].legend(loc=())
    #plt.subplots_adjust(left=0.065, right=0.97, top=0.96, bottom=0.065, wspace=0.14)
    #fig.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.95, hspace=0.35)
    plt.savefig(aggregation_plots_path / 'specific_key_places_rounds.pdf', bbox_inches='tight')


def plot_mistakes(mistakes_events: MetricAggregation, aggregation_plots_path: Path):
    if len([s for s in mistakes_events.per_event_median_df.columns if 'GameBot' in s]) > 0:
        y_ticks = [0, 75, 150]
    else:
        y_ticks = [0, 20, 40]
    plot_place_title_df(mistakes_events.per_event_median_df, 'Mistakes', aggregation_plots_path / 'mistakes.pdf',
                        'Rounds', y_ticks, mistakes_events.per_event_iqr_df, force_y_min_zero=True)


def create_latex_tables(offense_events: MetricAggregation, defense_events: MetricAggregation,
                        emd_no_filter: MetricAggregation, emd_only_kill: MetricAggregation,
                        lifetimes_no_filter: MetricAggregation, shots_per_kill_no_filter: MetricAggregation,
                        plots_path: Path):
    with open(plots_path / 'latex.txt', 'w') as f:
        f.write(f'''
\\begin{{tabular}}{{ r | r@{{\\hspace{{0em}}}}r r@{{\hspace{{0em}}}}r }} 
     & \multicolumn{{2}}{{r}}{{Offense}} &  \multicolumn{{2}}{{r}}{{Defense}} \\\\
    \\hline
    \\learnedbot & {offense_events.all_events_median_iqr_df.iloc[0, 0]*100:.0f}\\% $\pm$ & {offense_events.all_events_median_iqr_df.iloc[1, 0]*100:.0f}\\% & {defense_events.all_events_median_iqr_df.iloc[0, 0]*100:.0f}\\% $\pm$ & {defense_events.all_events_median_iqr_df.iloc[1, 0]*100:.0f}\\% \\\\
    \\hline
    \\handcraftedbot & {offense_events.all_events_median_iqr_df.iloc[0, 1]*100:.0f}\\% $\pm$ & {offense_events.all_events_median_iqr_df.iloc[1, 1]*100:.0f}\\% & {defense_events.all_events_median_iqr_df.iloc[0, 1]*100:.0f}\\% $\pm$ & {defense_events.all_events_median_iqr_df.iloc[1, 1]*100:.0f}\\% \\\\
    \\hline
    \\defaultbot & {offense_events.all_events_median_iqr_df.iloc[0, 2]*100:.0f}\\% $\pm$ & {offense_events.all_events_median_iqr_df.iloc[1, 2]*100:.0f}\\% & {defense_events.all_events_median_iqr_df.iloc[0, 2]*100:.0f}\\% $\pm$ & {defense_events.all_events_median_iqr_df.iloc[1, 2]*100:.0f}\\%
\\end{{tabular}}
        ''')

        f.write(f'''
\\begin{{tabular}}{{ r | r@{{\\hspace{{0.2em}}}}l r@{{\hspace{{0em}}}}r r@{{\\hspace{{0em}}}}r }} 
    EMD Type & \multicolumn{{2}}{{r}}{{\\learnedbot}} & \multicolumn{{2}}{{r}}{{\\handcraftedbot}} & \multicolumn{{2}}{{r}}{{\\defaultbot}} \\\\
    \\hline
    Map Occupancy & {emd_no_filter.per_event_median_df.iloc[0, 0]:.1f} $\pm$ & {emd_no_filter.per_event_iqr_df.iloc[0, 0]:.1f} & {emd_no_filter.per_event_median_df.iloc[1, 0]:.1f} $\pm$ & {emd_no_filter.per_event_iqr_df.iloc[1, 0]:.1f} & {emd_no_filter.per_event_median_df.iloc[2, 0]:.1f} $\pm$ & {emd_no_filter.per_event_iqr_df.iloc[2, 0]:.1f} \\\\
    \\hline
    Kill Locations & {emd_only_kill.per_event_median_df.iloc[0, 0]:.1f} $\pm$ & {emd_only_kill.per_event_iqr_df.iloc[0, 0]:.1f} & {emd_only_kill.per_event_median_df.iloc[1, 0]:.1f} $\pm$ & {emd_only_kill.per_event_iqr_df.iloc[1, 0]:.1f} & {emd_only_kill.per_event_median_df.iloc[2, 0]:.1f} $\pm$ & {emd_only_kill.per_event_iqr_df.iloc[2, 0]:.1f} \\\\
    \\hline
    Lifetimes & {lifetimes_no_filter.per_event_median_df.iloc[0, 0]:.1f} $\pm$ & {lifetimes_no_filter.per_event_iqr_df.iloc[0, 0]:.1f} & {lifetimes_no_filter.per_event_median_df.iloc[0, 1]:.1f} $\pm$ & {lifetimes_no_filter.per_event_iqr_df.iloc[0, 1]:.1f} & {lifetimes_no_filter.per_event_median_df.iloc[0, 2]:.1f} $\pm$ & {lifetimes_no_filter.per_event_iqr_df.iloc[0, 2]:.1f} \\\\
    \\hline
    Shots Per Kill & {shots_per_kill_no_filter.per_event_median_df.iloc[0, 0]:.1f} $\pm$ & {shots_per_kill_no_filter.per_event_iqr_df.iloc[0, 0]:.1f} & {shots_per_kill_no_filter.per_event_median_df.iloc[0, 1]:.1f} $\pm$ & {shots_per_kill_no_filter.per_event_iqr_df.iloc[0, 1]:.1f} & {shots_per_kill_no_filter.per_event_median_df.iloc[0, 2]:.1f} $\pm$ & {shots_per_kill_no_filter.per_event_iqr_df.iloc[0, 2]:.1f}
\\end{{tabular}}
        ''')



def aggregate_trajectory_metrics(rollout_extensions: list[str], rollout_prefix: str):
    aggregation_plots_path = similarity_plots_path / ("agg_" + rollout_prefix + rollout_extensions[0])
    aggregation_plots_path.mkdir(parents=True, exist_ok=True)

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    offense_events = aggregate_one_metric_type(rollout_extensions, rollout_prefix,
                                               Path("offense_flanks_pct_for_aggregation.csv"), True,
                                               aggregation_plots_path, 'offense_flanks')
    defense_events = aggregate_one_metric_type(rollout_extensions, rollout_prefix,
                                               Path("defense_spread_pct_for_aggregation.csv"), True,
                                               aggregation_plots_path, 'defense_spread')
    plot_offense_defense(offense_events, defense_events, aggregation_plots_path)

    mistakes_events = aggregate_one_metric_type(rollout_extensions, rollout_prefix,
                                                Path("mistakes_aggregation.csv"), True,
                                                aggregation_plots_path, 'mistakes')
    plot_mistakes(mistakes_events, aggregation_plots_path)

    emd_no_filter = \
        aggregate_one_metric_type(rollout_extensions, rollout_prefix, Path("diff") / "emd_no_filter.txt", False,
                                  aggregation_plots_path, 'emd_no_filter')
    emd_only_kill = aggregate_one_metric_type(rollout_extensions, rollout_prefix, Path("diff") / "emd_only_kill.txt", False,
                                              aggregation_plots_path, 'emd_only_kill')

    lifetimes_no_filter = \
        aggregate_one_metric_type(rollout_extensions, rollout_prefix, Path("lifetimes_no_filter.txt"), False,
                                  aggregation_plots_path, 'lifetimes_no_filter', 'Lifetimes', Path("diff") / "emd_no_filter.txt")
    shots_per_kill_no_filter = \
        aggregate_one_metric_type(rollout_extensions, rollout_prefix, Path("shots_per_kill_no_filter.txt"), False,
                                  aggregation_plots_path, 'shots_per_kill_no_filter', 'Shots Per Kill', Path("diff") / "emd_no_filter.txt")
    create_latex_tables(offense_events, defense_events, emd_no_filter, emd_only_kill,
                        lifetimes_no_filter, shots_per_kill_no_filter, aggregation_plots_path)


if __name__ == "__main__":
    rollout_extensions = sys.argv[1].split(',')
    rollout_prefix = sys.argv[2] + "_" if len(sys.argv) >= 3 else ""
    aggregate_trajectory_metrics(rollout_extensions, rollout_prefix)