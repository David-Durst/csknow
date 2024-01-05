from dataclasses import dataclass
import dataclasses
from math import isnan
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.order.column_names import c4_time_left_percent
from learn_bot.latent.place_area.column_names import get_similarity_column, specific_player_place_area_columns, \
    float_c4_cols, hdf5_id_columns
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.place_area.push_save_label import PushSaveRoundLabels
from learn_bot.latent.train_paths import default_save_push_round_labels_path
from learn_bot.latent.vis import run_vis_checkpoint
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper
from learn_bot.libs.io_transforms import flatten_list
from learn_bot.libs.multi_hdf5_wrapper import absolute_to_relative_train_test_key

fig_length = 6
error_col = 'Push/Save Label Error'
push_save_decile_col = 'Push/Save Predicted Label Decile'
predicted_similarity_col = 'Predicted Round Push/Save Label'
ground_truth_similarity_col = 'Ground Truth Round Push/Save Label'


class RoundSimilarityResults:
    num_not_labeled: int
    predicted_similarity_label: List[float]
    ground_truth_similarity_label: List[float]
    predicted_decile: List[float]
    abs_similarity_errors: List[float]

    def __init__(self):
        self.num_not_labeled = 0
        self.predicted_similarity_label = []
        self.ground_truth_similarity_label = []
        self.predicted_decile = []
        self.abs_similarity_errors = []


def leave_one_out_similarity_analysis(push_save_round_labels: PushSaveRoundLabels,
                                      hdf5_wrapper: HDF5Wrapper, test_group_ids: List[int]):
    # compute accuracy for all round ids, recording if in test or train set
    train_result = RoundSimilarityResults()
    test_result = RoundSimilarityResults()
    round_ids_and_similarity_df = \
        hdf5_wrapper.id_df.groupby(round_id_column, as_index=False)[get_similarity_column(0)].first()
    for _, round_id_and_similarity_row in round_ids_and_similarity_df.iterrows():
        round_id = round_id_and_similarity_row[round_id_column]
        is_test_round = round_id in test_group_ids
        predicted_similarity_label = round_id_and_similarity_row[get_similarity_column(0)]
        ground_truth_similarity_label = push_save_round_labels.round_id_to_data[round_id].to_float_label()
        similarity_error = abs(predicted_similarity_label - ground_truth_similarity_label)
        not_labeled = predicted_similarity_label < 0.
        if is_test_round:
            if not_labeled:
                test_result.num_not_labeled += 1
            else:
                test_result.predicted_similarity_label.append(predicted_similarity_label)
                test_result.ground_truth_similarity_label.append(ground_truth_similarity_label)
                test_result.abs_similarity_errors.append(similarity_error)
                test_result.predicted_decile.append(int(predicted_similarity_label * 10.) / 10.)
        else:
            if not_labeled:
                train_result.num_not_labeled += 1
            else:
                train_result.predicted_similarity_label.append(predicted_similarity_label)
                train_result.ground_truth_similarity_label.append(ground_truth_similarity_label)
                train_result.abs_similarity_errors.append(similarity_error)
                train_result.predicted_decile.append(int(predicted_similarity_label * 10.) / 10.)

    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # plot two histograms
    train_result_df = pd.DataFrame.from_dict({
        predicted_similarity_col: train_result.predicted_similarity_label,
        ground_truth_similarity_col: train_result.ground_truth_similarity_label,
        push_save_decile_col: train_result.predicted_decile,
        error_col: train_result.abs_similarity_errors
    })
    test_result_df = pd.DataFrame.from_dict({
        predicted_similarity_col: test_result.predicted_similarity_label,
        ground_truth_similarity_col: test_result.ground_truth_similarity_label,
        push_save_decile_col: test_result.predicted_decile,
        error_col: test_result.abs_similarity_errors
    })
    bins = [i * 0.1 for i in range(11)]
    fig = plt.figure(figsize=(5*fig_length, fig_length), constrained_layout=True)
    axs = fig.subplots(1, 5, squeeze=False)
    train_errors_and_push_decile_df_splits = [
        train_result_df,
        train_result_df[train_result_df[push_save_decile_col] == 0.],
        train_result_df[train_result_df[push_save_decile_col].between(0., 0.5, inclusive='neither')],
        train_result_df[train_result_df[push_save_decile_col].between(0.5, 1.0, inclusive='left')],
        train_result_df[train_result_df[push_save_decile_col] == 1.]
    ]
    test_errors_and_push_decile_df_splits = [
        test_result_df,
        test_result_df[test_result_df[push_save_decile_col] == 0.],
        test_result_df[test_result_df[push_save_decile_col].between(0., 0.5, inclusive='neither')],
        test_result_df[test_result_df[push_save_decile_col].between(0.5, 1.0, inclusive='left')],
        test_result_df[test_result_df[push_save_decile_col] == 1.]
    ]
    split_names = ['', ' Label == 0', ' 0 < Label < 0.5', ' 0.5 <= Label < 1.', ' Label == 1']

    for i in range(len(train_errors_and_push_decile_df_splits)):
        axs[0, i].set_title('Push/Save Label Error Distribution' + split_names[i])
        axs[0, i].set_xlabel(error_col)
        axs[0, i].set_ylabel('Percent Rounds')

        train_abs_similarity_errors_series = train_errors_and_push_decile_df_splits[i][error_col]
        train_abs_similarity_errors_np = train_abs_similarity_errors_series.values
        test_abs_similarity_errors_series = test_errors_and_push_decile_df_splits[i][error_col]
        test_abs_similarity_errors_np = test_abs_similarity_errors_series.values

        train_weights = np.ones(len(train_abs_similarity_errors_np)) / len(train_abs_similarity_errors_np)
        test_weights = np.ones(len(test_abs_similarity_errors_np)) / len(test_abs_similarity_errors_np)
        axs[0, i].hist([train_abs_similarity_errors_np, test_abs_similarity_errors_np], bins,
                histtype='bar', label=['Train', 'Test'], weights=[train_weights, test_weights])
        axs[0, i].set_ylim(0., 1.)
        train_description = 'Train\n' + train_abs_similarity_errors_series.describe().to_string() + \
                            f'\n not labeled {train_result.num_not_labeled} '
        axs[0, i].text(0.2, 0.2, train_description, family='monospace')
        test_description = 'Test\n' + test_abs_similarity_errors_series.describe().to_string() + \
                            f'\n not labeled {test_result.num_not_labeled} '
        axs[0, i].text(0.65, 0.2, test_description, family='monospace')
        axs[0, i].legend()

    plt.savefig(similarity_plots_path / 'similarity_errors.png')

    fig = plt.figure(figsize=(2*fig_length, fig_length), constrained_layout=True)
    axs = fig.subplots(1, 2, squeeze=False)
    train_result_df.plot.scatter(ground_truth_similarity_col, predicted_similarity_col, ax=axs[0, 0])
    axs[0, 0].set_ylim(0., 1.)
    axs[0, 0].set_xlim(0., 1.)
    axs[0, 0].set_title('Train Ground Truth vs Predicted Push/Save Labels')
    train_description = 'Train Errors\n' + train_result_df[error_col].describe().to_string() + \
                        f'\n not labeled {train_result.num_not_labeled} '
    axs[0, 0].text(0.2, 0.2, train_description, family='monospace')
    test_result_df.plot.scatter(ground_truth_similarity_col, predicted_similarity_col, ax=axs[0, 1])
    axs[0, 1].set_ylim(0., 1.)
    axs[0, 1].set_xlim(0., 1.)
    axs[0, 1].set_title('Test Ground Truth vs Predicted Push/Save Labels')
    test_description = 'Test Errors\n' + test_result_df[error_col].describe().to_string() + \
                       f'\n not labeled {test_result.num_not_labeled} '
    axs[0, 1].text(0.2, 0.2, test_description, family='monospace')

    plt.savefig(similarity_plots_path / 'similarity_scatter.png')


last_c4_time_percent_col = "last C4 time percent in round"


class TickSimilarityResults:
    num_not_labeled: int
    player_tick_label_5s: List[bool]
    player_tick_label_10s: List[bool]
    player_tick_label_20s: List[bool]
    player_tick_label_similarity: List[bool]
    player_tick_label_ground_truth: List[bool]

    def __init__(self):
        self.num_not_labeled = 0
        self.player_tick_label_5s = []
        self.player_tick_label_10s = []
        self.player_tick_label_20s = []
        self.player_tick_label_similarity = []
        self.player_tick_label_ground_truth = []


def per_tick_similarity_analysis(push_save_round_labels: PushSaveRoundLabels,
                                 hdf5_wrapper: HDF5Wrapper, test_group_ids: List[int]):
    decrease_distance_to_c4_5s_cols = [player_place_area_columns.decrease_distance_to_c4_5s
                                       for player_place_area_columns in specific_player_place_area_columns]
    decrease_distance_to_c4_10s_cols = [player_place_area_columns.decrease_distance_to_c4_10s
                                        for player_place_area_columns in specific_player_place_area_columns]
    decrease_distance_to_c4_20s_cols = [player_place_area_columns.decrease_distance_to_c4_20s
                                        for player_place_area_columns in specific_player_place_area_columns]
    alive_cols = [player_place_area_columns.alive for player_place_area_columns in specific_player_place_area_columns]
    cols_to_get = alive_cols + decrease_distance_to_c4_5s_cols + decrease_distance_to_c4_10s_cols + \
        decrease_distance_to_c4_20s_cols + float_c4_cols + hdf5_id_columns
    df = load_hdf5_to_pd(hdf5_wrapper.hdf5_path, cols_to_get=cols_to_get)

    id_df = hdf5_wrapper.id_df.copy()
    id_df[last_c4_time_percent_col] = df[c4_time_left_percent[0]]
    round_ids_similarity_last_c4_time_df = \
        id_df.groupby(round_id_column, as_index=False)[[get_similarity_column(0), last_c4_time_percent_col]].last()
    df = df.merge(round_ids_similarity_last_c4_time_df, on=round_id_column)

    train_result = TickSimilarityResults()
    test_result = TickSimilarityResults()

    with tqdm(total=len(df), disable=False) as pbar:
        for _, round_row in df.iterrows():
            round_id = round_row[round_id_column]
            fraction_of_round = (1. - round_row[c4_time_left_percent]) / (1. - round_row[last_c4_time_percent_col])
            is_test_round = round_id in test_group_ids

            ground_truth_round_similarity_label = push_save_round_labels.round_id_to_data[round_id].to_float_label()
            ground_truth_tick_similarity_label = (fraction_of_round <= ground_truth_round_similarity_label).iloc[0]

            predicted_round_similarity_label = round_row[get_similarity_column(0)]
            predicted_tick_similarity_label = (fraction_of_round <= predicted_round_similarity_label).iloc[0]

            player_tick_label_5s_alive_and_dead = list(round_row[decrease_distance_to_c4_5s_cols] > 0.5)
            player_tick_label_10s_alive_and_dead = list(round_row[decrease_distance_to_c4_10s_cols] > 0.5)
            player_tick_label_20s_alive_and_dead = list(round_row[decrease_distance_to_c4_20s_cols] > 0.5)

            # need to filter only to labels for players that are alive
            alive_player_indices = [i for i, alive in enumerate(list(round_row[alive_cols])) if alive]
            player_tick_label_5s = []
            player_tick_label_10s = []
            player_tick_label_20s = []
            for i in alive_player_indices:
                player_tick_label_5s.append(player_tick_label_5s_alive_and_dead[i])
                player_tick_label_10s.append(player_tick_label_10s_alive_and_dead[i])
                player_tick_label_20s.append(player_tick_label_20s_alive_and_dead[i])
            player_tick_label_similarity = [predicted_tick_similarity_label for _ in player_tick_label_5s]
            player_tick_label_ground_truth = [ground_truth_tick_similarity_label for _ in player_tick_label_5s]

            if predicted_round_similarity_label < 0:
                if is_test_round:
                    test_result.num_not_labeled += len(player_tick_label_5s)
                else:
                    train_result.num_not_labeled += len(player_tick_label_5s)
                continue

            if is_test_round:
                result_to_update = test_result
            else:
                result_to_update = train_result
            result_to_update.player_tick_label_5s += player_tick_label_5s
            result_to_update.player_tick_label_10s += player_tick_label_10s
            result_to_update.player_tick_label_20s += player_tick_label_20s
            result_to_update.player_tick_label_similarity += player_tick_label_similarity
            result_to_update.player_tick_label_ground_truth += player_tick_label_ground_truth
            pbar.update(1)


    fig = plt.figure(figsize=(4*fig_length, 2*fig_length), constrained_layout=True)
    fig.suptitle('Per-Tick Push/Save Labels')
    axs = fig.subplots(2, 4, squeeze=False)
    ConfusionMatrixDisplay.from_predictions(train_result.player_tick_label_similarity,
                                            train_result.player_tick_label_ground_truth,
                                            display_labels=['Push', 'Save'], ax=axs[0, 0], normalize='all')
    axs[0, 0].set_title('Train Nearest Neighbor')
    ConfusionMatrixDisplay.from_predictions(train_result.player_tick_label_5s,
                                            train_result.player_tick_label_ground_truth,
                                            display_labels=['Push', 'Save'], ax=axs[0, 1], normalize='all')
    axs[0, 1].set_title('Train 5s Distance Decrease')
    ConfusionMatrixDisplay.from_predictions(train_result.player_tick_label_10s,
                                            train_result.player_tick_label_ground_truth,
                                            display_labels=['Push', 'Save'], ax=axs[0, 2], normalize='all')
    axs[0, 2].set_title('Train 10s Distance Decrease')
    ConfusionMatrixDisplay.from_predictions(train_result.player_tick_label_20s,
                                            train_result.player_tick_label_ground_truth,
                                            display_labels=['Push', 'Save'], ax=axs[0, 3], normalize='all')
    axs[0, 3].set_title('Train 20s Distance Decrease')
    ConfusionMatrixDisplay.from_predictions(test_result.player_tick_label_similarity,
                                            test_result.player_tick_label_ground_truth,
                                            display_labels=['Push', 'Save'], ax=axs[1, 0], normalize='all')
    axs[1, 0].set_title('Test Nearest Neighbor')
    ConfusionMatrixDisplay.from_predictions(test_result.player_tick_label_5s,
                                            test_result.player_tick_label_ground_truth,
                                            display_labels=['Push', 'Save'], ax=axs[1, 1], normalize='all')
    axs[1, 1].set_title('Test 5s Distance Decrease')
    ConfusionMatrixDisplay.from_predictions(test_result.player_tick_label_10s,
                                            test_result.player_tick_label_ground_truth,
                                            display_labels=['Push', 'Save'], ax=axs[1, 2], normalize='all')
    axs[1, 2].set_title('Test 10s Distance Decrease')
    ConfusionMatrixDisplay.from_predictions(test_result.player_tick_label_20s,
                                            test_result.player_tick_label_ground_truth,
                                            display_labels=['Push', 'Save'], ax=axs[1, 3], normalize='all')
    axs[1, 3].set_title('Test 20s Distance Decrease')

    plt.savefig(similarity_plots_path / 'similarity_confusionq.png')
    #train_5s_accuracy = sum(train_result.correct_player_tick_label_5s) / len(train_result.correct_player_tick_label_5s)
    #train_10s_accuracy = sum(train_result.correct_player_tick_label_10s) / len(train_result.correct_player_tick_label_10s)
    #train_20s_accuracy = sum(train_result.correct_player_tick_label_20s) / len(train_result.correct_player_tick_label_20s)
    #train_similarity_accuracy = sum(train_result.correct_player_tick_label_similarity) / len(train_result.correct_player_tick_label_similarity)
    #print(f"train 5s accuracy {train_5s_accuracy:.2f} train 10s accuracy {train_10s_accuracy:.2f} "
    #      f"train 20s accuracy {train_20s_accuracy:.2f} train similarity accuracy {train_similarity_accuracy:.2f}")

    #test_5s_accuracy = sum(test_result.correct_player_tick_label_5s) / len(test_result.correct_player_tick_label_5s)
    #test_10s_accuracy = sum(test_result.correct_player_tick_label_10s) / len(test_result.correct_player_tick_label_10s)
    #test_20s_accuracy = sum(test_result.correct_player_tick_label_20s) / len(test_result.correct_player_tick_label_20s)
    #test_similarity_accuracy = sum(test_result.correct_player_tick_label_similarity) / len(test_result.correct_player_tick_label_similarity)
    #print(f"test 5s accuracy {test_5s_accuracy:.2f} test 10s accuracy {test_10s_accuracy:.2f} "
    #      f"test 20s accuracy {test_20s_accuracy:.2f} test similarity accuracy {test_similarity_accuracy:.2f}")


if __name__ == "__main__":
    load_data_options = dataclasses.replace(run_vis_checkpoint.load_data_options, similarity_analysis=True)
    load_data_result = LoadDataResult(load_data_options=load_data_options)

    push_save_round_labels: PushSaveRoundLabels = PushSaveRoundLabels()
    push_save_round_labels.load(default_save_push_round_labels_path)

    # labels are on first hdf5 file
    hdf5_wrapper = load_data_result.multi_hdf5_wrapper.hdf5_wrappers[0]
    hdf5_key = absolute_to_relative_train_test_key(hdf5_wrapper.hdf5_path)
    test_group_ids = load_data_result.multi_hdf5_wrapper.test_group_ids[hdf5_key]
    # relying on 28th file to come first, which always has been true
    assert '_28.hdf5' in str(hdf5_key)

    leave_one_out_similarity_analysis(push_save_round_labels, hdf5_wrapper, test_group_ids)
    per_tick_similarity_analysis(push_save_round_labels, hdf5_wrapper, test_group_ids)
