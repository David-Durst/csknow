from typing import List

import pandas as pd
from learn_bot.latent.analyze.comparison_column_names import *
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from learn_bot.latent.place_area.load_data import LoadDataOptions
from learn_bot.latent.vis.vis_two import PredictedToGroundTruthDict, PredictedToGroundTruthRoundData


@dataclass
class ComparisonConfig:
    similarity_data_path: Path
    predicted_load_data_options: LoadDataOptions
    ground_truth_load_data_options: LoadDataOptions
    limit_predicted_df_to_bot_good: bool
    limit_predicted_df_to_human_good: bool
    metric_cost_file_name: str
    metric_cost_title: str


def generate_bins(min_bin_start: int, max_bin_end: int, bin_width: int) -> List[int]:
    return [b for b in range(min_bin_start, max_bin_end + bin_width, bin_width)]


def plot_hist(ax: plt.Axes, data: pd.Series, bins: List[int]):
    ax.hist(data.values, bins=bins, weights=np.ones(len(data)) / len(data))
    ax.grid(visible=True)
    # ax.yaxis.set_major_formatter(PercentFormatter(1))


dtw_cost_bins = generate_bins(0, 15000, 1000)
delta_distance_bins = generate_bins(-20000, 20000, 2500)
delta_time_bins = generate_bins(-40, 40, 5)


def plot_trajectory_comparison_histograms(similarity_df: pd.DataFrame, config: ComparisonConfig,
                                          similarity_plots_path: Path):
    pd.options.display.max_columns = None

    pd.options.display.max_rows = None
    pd.options.display.width = 1000
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    # print(similarity_df.loc[:, [predicted_round_id_col, best_fit_ground_truth_round_id_col, metric_type_col,
    #                            dtw_cost_col, delta_distance_col, delta_time_col]])

    # plot cost, distance, and time by metric type
    metric_types = similarity_df[metric_type_col].unique().tolist()
    metric_types_similarity_df = similarity_df.loc[:,
                                 [metric_type_col, dtw_cost_col, delta_distance_col, delta_time_col]]

    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    fig.suptitle(config.metric_cost_title)
    axs = fig.subplots(len(metric_types), 3, squeeze=False)
    for i, metric_type in enumerate(metric_types):
        metric_type_str = metric_type.decode('utf-8')
        metric_type_similarity_df = metric_types_similarity_df[(similarity_df[metric_type_col] == metric_type)]
        plot_hist(axs[i, 0], metric_type_similarity_df[dtw_cost_col], dtw_cost_bins)
        axs[i, 0].set_title(metric_type_str + " DTW Cost")
        axs[i, 0].set_ylim(0., 0.6)
        axs[i, 0].text(5000, 0.4, metric_type_similarity_df[dtw_cost_col].describe().to_string(), family='monospace')
        plot_hist(axs[i, 1], metric_type_similarity_df[delta_distance_col], delta_distance_bins)
        axs[i, 1].set_title(metric_type_str + " Delta Distance")
        axs[i, 1].set_ylim(0., 0.6)
        axs[i, 1].text(0, 0.4, metric_type_similarity_df[delta_distance_col].describe().to_string(), family='monospace')
        plot_hist(axs[i, 2], metric_type_similarity_df[delta_time_col], delta_time_bins)
        axs[i, 2].set_title(metric_type_str + " Delta Time")
        axs[i, 2].set_ylim(0., 0.6)
        axs[i, 2].text(0, 0.4, metric_type_similarity_df[delta_time_col].describe().to_string(), family='monospace')
    plt.savefig(similarity_plots_path / (config.metric_cost_file_name + '.png'))


def build_predicted_to_ground_truth_dict(similarity_df: pd.DataFrame) -> PredictedToGroundTruthDict:
    predicted_to_ground_truth_dict: PredictedToGroundTruthDict = {}
    for idx, row in similarity_df.iterrows():
        metric_type = row[metric_type_col].decode('utf-8')
        agent_mapping_str = row[agent_mapping_col].decode('utf-8')
        agent_mapping = {}
        for agent_pair in agent_mapping_str.split(','):
            agents = [int(agent) for agent in agent_pair.split('_')]
            agent_mapping[int(agents[0])] = int(agents[1])
        if row[predicted_trace_batch_col] not in predicted_to_ground_truth_dict:
            predicted_to_ground_truth_dict[row[predicted_trace_batch_col]] = {}
        if row[predicted_round_id_col] not in predicted_to_ground_truth_dict[row[predicted_trace_batch_col]]:
            predicted_to_ground_truth_dict[row[predicted_trace_batch_col]][row[predicted_round_id_col]] = {}
        if metric_type not in \
                predicted_to_ground_truth_dict[row[predicted_trace_batch_col]][row[predicted_round_id_col]]:
            predicted_to_ground_truth_dict[row[predicted_trace_batch_col]][row[predicted_round_id_col]][metric_type] = []
        predicted_to_ground_truth_dict[row[predicted_trace_batch_col]][row[predicted_round_id_col]][metric_type].append(
            PredictedToGroundTruthRoundData(row[predicted_round_id_col], row[best_fit_ground_truth_round_id_col],
                                            row[best_fit_ground_truth_trace_batch_col],
                                            row, agent_mapping))
    return predicted_to_ground_truth_dict
