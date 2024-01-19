from typing import List, Dict, Tuple, Union

import pandas as pd
from matplotlib.ticker import PercentFormatter

from learn_bot.latent.analyze.comparison_column_names import *
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from learn_bot.latent.analyze.create_test_plant_states import hdf5_key_column, push_only_test_plant_states_file_name, \
    save_only_test_plant_states_file_name
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.place_area.column_names import get_similarity_column
from learn_bot.latent.place_area.load_data import LoadDataOptions, LoadDataResult
from learn_bot.latent.vis.vis_two import PredictedToGroundTruthDict, PredictedToGroundTruthRoundData
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.multi_hdf5_wrapper import train_test_split_folder_path, absolute_to_relative_train_test_key
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


def generate_bins(min_bin_start: int, max_bin_end: int, bin_width: int) -> List[int]:
    return [b for b in range(min_bin_start, max_bin_end + bin_width, bin_width)]


def plot_hist(ax: plt.Axes, data: pd.Series, bins: List[Union[int,float]]):
    ax.hist(data.values, bins=bins, weights=np.ones(len(data)) / len(data), color="#3f8f35", edgecolor="#3f8f35")
    ax.grid(visible=True)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def percentile_filter_series(data: pd.Series, low_pct_to_remove=0.01, high_pct_to_remove=0.01) -> pd.Series:
    q_low = data.quantile(low_pct_to_remove)
    q_hi = data.quantile(1. - high_pct_to_remove)
    return data[(data <= q_hi) & (data >= q_low)]

dtw_cost_bins = generate_bins(0, 16000, 250)
delta_distance_bins = generate_bins(-20000, 20000, 2500)
delta_time_bins = generate_bins(-40, 40, 5)


def get_test_plant_states_pd(push_only: bool = True) -> pd.DataFrame:
    test_plant_states_path = \
        train_test_split_folder_path / (push_only_test_plant_states_file_name if push_only
                                        else save_only_test_plant_states_file_name)
    test_start_pd = load_hdf5_to_pd(test_plant_states_path)
    test_start_pd[hdf5_key_column] = test_start_pd[hdf5_key_column].str.decode('utf-8')
    return test_start_pd


def get_hdf5_to_test_round_ids(push_only: bool = True) -> Tuple[Dict[str, List[int]], Dict[Path, List[int]]]:
    test_plant_states_path = \
        train_test_split_folder_path / (push_only_test_plant_states_file_name if push_only
                                        else save_only_test_plant_states_file_name)
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


# above function loads from a file to get exactly what is run in game, this recomputes and allows more flexibility
def get_hdf5_to_round_ids_fresh(load_data_result: LoadDataResult, test_only: bool,
                                push_only: bool = True, save_only: bool = False) -> Tuple[Dict[str, List[int]], Dict[Path, List[int]]]:
    hdf5_wrappers = load_data_result.multi_hdf5_wrapper.test_hdf5_wrappers if test_only else\
        load_data_result.multi_hdf5_wrapper.hdf5_wrappers
    # convert keys to partial keys used in similarity_df, get round ids for each hdf5
    hdf5_partial_key_to_round_ids: Dict[str, List[int]] = {}
    hdf5_key_to_round_ids: Dict[Path, List[int]] = {}
    for hdf5_wrapper in hdf5_wrappers:
        if push_only or save_only:
            push_similarity_column = get_similarity_column(0)
            assert push_similarity_column in hdf5_wrapper.id_df.columns
            round_and_similarity_df = hdf5_wrapper.id_df.groupby(round_id_column,
                                                                 as_index=False)[push_similarity_column].first()
            if push_only:
                push_save_condition = round_and_similarity_df[push_similarity_column] > 0.5
            else:
                push_save_condition = round_and_similarity_df[push_similarity_column] < 0.5
            round_ids = list(round_and_similarity_df[push_save_condition][round_id_column])
        else:
            round_ids = list(hdf5_wrapper.id_df[round_id_column].unique())
        hdf5_partial_key = str(hdf5_wrapper.hdf5_path.name)
        hdf5_partial_key_to_round_ids[hdf5_partial_key] = round_ids
        hdf5_key = absolute_to_relative_train_test_key(hdf5_wrapper.hdf5_path)
        hdf5_key_to_round_ids[hdf5_key] = round_ids
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
    #for _, v in predicted_to_ground_truth_dict.items():
    #    for vk, vv in v.items():
    #        if 'Slope Constrained DTW' in vv and len(vv['Unconstrained DTW']) == 5 and len(vv['Slope Constrained DTW']) == 1:
    #            print('Unconstrained Match But only one Slope Constrained')
    return predicted_to_ground_truth_dict

