from typing import List, Dict, Tuple

import pandas as pd
from learn_bot.latent.analyze.comparison_column_names import *
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from learn_bot.latent.analyze.create_test_plant_states import hdf5_key_column, push_only_test_plant_states_file_name
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.place_area.load_data import LoadDataOptions, LoadDataResult
from learn_bot.latent.vis.vis_two import PredictedToGroundTruthDict, PredictedToGroundTruthRoundData
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.pd_printing import set_pd_print_options


@dataclass
class ComparisonConfig:
    similarity_data_path: Path
    predicted_load_data_options: LoadDataOptions
    ground_truth_load_data_options: LoadDataOptions
    limit_predicted_df_to_bot_good: bool
    limit_predicted_df_to_human_good: bool
    metric_cost_file_name: str
    metric_cost_title: str
    limit_predicted_to_first_n_test_rounds: bool = False


def generate_bins(min_bin_start: int, max_bin_end: int, bin_width: int) -> List[int]:
    return [b for b in range(min_bin_start, max_bin_end + bin_width, bin_width)]


def plot_hist(ax: plt.Axes, data: pd.Series, bins: List[int]):
    ax.hist(data.values, bins=bins, weights=np.ones(len(data)) / len(data))
    ax.grid(visible=True)
    # ax.yaxis.set_major_formatter(PercentFormatter(1))


def percentile_filter_series(data: pd.Series, low_pct_to_remove=0.01, high_pct_to_remove=0.01) -> pd.Series:
    q_low = data.quantile(low_pct_to_remove)
    q_hi = data.quantile(1. - high_pct_to_remove)
    return data[(data <= q_hi) & (data >= q_low)]

dtw_cost_bins = generate_bins(0, 8000, 250)
delta_distance_bins = generate_bins(-20000, 20000, 2500)
delta_time_bins = generate_bins(-40, 40, 5)


def filter_similarity_for_first_n_test_rounds(loaded_data_result: LoadDataResult, similarity_df: pd.DataFrame,
                                              top_n: int = 300) -> pd.DataFrame:
    hdf5_partial_key_to_round_ids, _ = get_hdf5_to_round_ids(loaded_data_result)

    # build condition on similarity df
    # start false so can build up ors
    valid_hdf5_and_round_id_condition = \
        similarity_df[predicted_trace_batch_col] != similarity_df[predicted_trace_batch_col]
    predicted_trace_batch_series = similarity_df[predicted_trace_batch_col].str.decode('utf-8')
    for hdf5_partial_key, round_ids in hdf5_partial_key_to_round_ids.items():
        valid_hdf5_and_round_id_condition = valid_hdf5_and_round_id_condition | \
                                            ((predicted_trace_batch_series == hdf5_partial_key) &
                                             (similarity_df[predicted_round_id_col].isin(round_ids)))
        #for round_id in round_ids:
        #    if len(similarity_df[(predicted_trace_batch_series == hdf5_partial_key) & (similarity_df[predicted_round_id_col] == round_id)]) == 0:
        #        print('test round without match in similarity df')

    return similarity_df[valid_hdf5_and_round_id_condition]


def get_hdf5_to_round_ids(loaded_data_result) -> Tuple[Dict[str, List[int]], Dict[Path, List[int]]]:
    test_plant_states_path = \
        loaded_data_result.multi_hdf5_wrapper.train_test_split_path.parent / push_only_test_plant_states_file_name
    test_start_pd = load_hdf5_to_pd(test_plant_states_path)  # .iloc[:top_n]
    # convert keys to partial keys used in similarity_df, get round ids for each hdf5
    hdf5_partial_key_to_round_ids: Dict[str, List[int]] = {}
    hdf5_key_to_round_ids: Dict[Path, List[int]] = {}
    hdf5_keys_series = test_start_pd[hdf5_key_column].str.decode('utf-8')
    total_round_ids = 0
    for hdf5_key in hdf5_keys_series.unique():
        hdf5_partial_key = str(Path(hdf5_key).name)
        round_ids = list(test_start_pd[hdf5_keys_series == hdf5_key][round_id_column].unique())
        hdf5_partial_key_to_round_ids[hdf5_partial_key] = round_ids
        hdf5_key_to_round_ids[Path(hdf5_key)] = round_ids
        total_round_ids += len(round_ids)
    return hdf5_partial_key_to_round_ids, hdf5_key_to_round_ids


def plot_trajectory_comparison_histograms(similarity_df: pd.DataFrame, config: ComparisonConfig,
                                          similarity_plots_path: Path):
    set_pd_print_options()
    # print(similarity_df.loc[:, [predicted_round_id_col, best_fit_ground_truth_round_id_col, metric_type_col,
    #                            dtw_cost_col, delta_distance_col, delta_time_col]])

    # plot cost, distance, and time by metric type
    metric_types = similarity_df[metric_type_col].unique().tolist()
    metric_types_similarity_df = similarity_df.loc[:,
                                 [metric_type_col, dtw_cost_col, delta_distance_col, delta_time_col, best_match_id_col]]

    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    fig.suptitle(config.metric_cost_title)
    axs = fig.subplots(len(metric_types), 3, squeeze=False)
    for i, metric_type in enumerate(metric_types):
        metric_type_str = metric_type.decode('utf-8')
        metric_type_similarity_df = metric_types_similarity_df[(metric_types_similarity_df[metric_type_col] == metric_type)]# &
                                                               #(metric_types_similarity_df[best_match_id_col] < 2)]
        plot_hist(axs[i, 0], metric_type_similarity_df[dtw_cost_col], dtw_cost_bins)
        axs[i, 0].set_title(metric_type_str + " DTW Cost")
        axs[i, 0].set_ylim(0., 0.6)
        dtw_description = metric_type_similarity_df[dtw_cost_col].describe()
        if 'Slope' in metric_type_str:
            with open(similarity_plots_path / (config.metric_cost_file_name + '.txt'), 'w') as cost_f:
                cost_f.write(f"{config.metric_cost_title} & {dtw_description['mean']:.2f} & {dtw_description['std']:.2f} \\\\ \n")
        axs[i, 0].text(5000, 0.4, dtw_description.to_string(), family='monospace')
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
        predicted_trace_batch = row[predicted_trace_batch_col].decode('utf-8')
        metric_type = row[metric_type_col].decode('utf-8')
        agent_mapping_str = row[agent_mapping_col].decode('utf-8')
        agent_mapping = {}
        for agent_pair in agent_mapping_str.split(','):
            agents = [int(agent) for agent in agent_pair.split('_')]
            agent_mapping[int(agents[0])] = int(agents[1])
        if predicted_trace_batch not in predicted_to_ground_truth_dict:
            predicted_to_ground_truth_dict[predicted_trace_batch] = {}
        if row[predicted_round_id_col] not in predicted_to_ground_truth_dict[predicted_trace_batch]:
            predicted_to_ground_truth_dict[predicted_trace_batch][row[predicted_round_id_col]] = {}
        if metric_type not in predicted_to_ground_truth_dict[predicted_trace_batch][row[predicted_round_id_col]]:
            predicted_to_ground_truth_dict[predicted_trace_batch][row[predicted_round_id_col]][metric_type] = []
        predicted_to_ground_truth_dict[predicted_trace_batch][row[predicted_round_id_col]][metric_type].append(
            PredictedToGroundTruthRoundData(row[predicted_round_id_col], row[best_fit_ground_truth_round_id_col],
                                            row[best_fit_ground_truth_trace_batch_col].decode('utf-8'),
                                            row, agent_mapping))
    return predicted_to_ground_truth_dict

