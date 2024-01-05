from dataclasses import dataclass
import dataclasses
from math import isnan
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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


def leave_one_out_similarity_analysis(push_save_round_labels: PushSaveRoundLabels,
                                      hdf5_wrapper: HDF5Wrapper, test_group_ids: List[int]):
    # compute accuracy for all round ids, recording if in test or train set
    num_train_not_labeled = 0
    train_abs_similarity_errors = []
    train_push_decile = []
    num_test_not_labeled = 0
    test_abs_similarity_errors = []
    test_push_decile = []
    round_ids_and_similarity_df = \
        hdf5_wrapper.id_df.groupby(round_id_column, as_index=False)[get_similarity_column(0)].first()
    for _, round_id_and_similarity_row in round_ids_and_similarity_df.iterrows():
        round_id = round_id_and_similarity_row[round_id_column]
        is_test_round = round_id in test_group_ids
        predicted_similarity_label = round_id_and_similarity_row[get_similarity_column(0)]
        similarity_error = abs(predicted_similarity_label -
                               push_save_round_labels.round_id_to_data[round_id].to_float_label())
        not_labeled = predicted_similarity_label < 0.
        if is_test_round:
            if not_labeled:
                num_test_not_labeled += 1
            else:
                test_abs_similarity_errors.append(similarity_error)
                test_push_decile.append(int(predicted_similarity_label * 10.) / 10.)
        else:
            if not_labeled:
                num_train_not_labeled += 1
            else:
                train_abs_similarity_errors.append(similarity_error)
                train_push_decile.append(int(predicted_similarity_label * 10.) / 10.)

    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # plot two histograms
    train_errors_and_push_decile_df = pd.DataFrame.from_dict({error_col: train_abs_similarity_errors,
                                                              push_save_decile_col: train_push_decile})
    test_errors_and_push_decile_df = pd.DataFrame.from_dict({error_col: test_abs_similarity_errors,
                                                             push_save_decile_col: test_push_decile})
    bins = [i * 0.1 for i in range(11)]
    fig = plt.figure(figsize=(5*fig_length, fig_length), constrained_layout=True)
    axs = fig.subplots(1, 5, squeeze=False)
    train_errors_and_push_decile_df_splits = [
        train_errors_and_push_decile_df,
        train_errors_and_push_decile_df[train_errors_and_push_decile_df[push_save_decile_col] == 0.],
        train_errors_and_push_decile_df[
            train_errors_and_push_decile_df[push_save_decile_col].between(0., 0.5, inclusive='neither')],
        train_errors_and_push_decile_df[
            train_errors_and_push_decile_df[push_save_decile_col].between(0.5, 1.0, inclusive='left')],
        train_errors_and_push_decile_df[train_errors_and_push_decile_df[push_save_decile_col] == 1.]
    ]
    test_errors_and_push_decile_df_splits = [
        test_errors_and_push_decile_df,
        test_errors_and_push_decile_df[test_errors_and_push_decile_df[push_save_decile_col] == 0.],
        test_errors_and_push_decile_df[
            test_errors_and_push_decile_df[push_save_decile_col].between(0., 0.5, inclusive='neither')],
        test_errors_and_push_decile_df[
            test_errors_and_push_decile_df[push_save_decile_col].between(0.5, 1.0, inclusive='left')],
        test_errors_and_push_decile_df[test_errors_and_push_decile_df[push_save_decile_col] == 1.]
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
                            f'\n not labeled {num_train_not_labeled} '
        axs[0, i].text(0.2, 0.2, train_description, family='monospace')
        test_description = 'Test\n' + test_abs_similarity_errors_series.describe().to_string() + \
                            f'\n not labeled {num_test_not_labeled} '
        axs[0, i].text(0.65, 0.2, test_description, family='monospace')
        axs[0, i].legend()

    plt.savefig(similarity_plots_path / 'similarity_errors.png')


last_c4_time_percent_col = "last C4 time percent in round"


class TickSimilarityResults:
    num_not_labeled: int
    correct_player_tick_label_5s: List[bool]
    correct_player_tick_label_10s: List[bool]
    correct_player_tick_label_20s: List[bool]
    correct_player_tick_label_similarity: List[bool]
    predicted_round_label_similarity_duplicated_per_player_tick: List[float]

    def __init__(self):
        self.num_not_labeled = 0
        self.correct_player_tick_label_5s = []
        self.correct_player_tick_label_10s = []
        self.correct_player_tick_label_20s = []
        self.correct_player_tick_label_similarity = []
        self.predicted_round_label_similarity_duplicated_per_player_tick = []


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
            ground_truth_tick_push_label = (fraction_of_round <= ground_truth_round_similarity_label).iloc[0]

            predicted_round_similarity_label = round_row[get_similarity_column(0)]
            predicted_tick_push_label = (fraction_of_round <= predicted_round_similarity_label).iloc[0]

            correct_player_tick_label_5s_alive_and_dead = list(
                (round_row[decrease_distance_to_c4_5s_cols] > 0.5) == ground_truth_tick_push_label
            )
            correct_player_tick_label_10s_alive_and_dead = list(
                (round_row[decrease_distance_to_c4_10s_cols] > 0.5) == ground_truth_tick_push_label
            )
            correct_player_tick_label_20s_alive_and_dead = list(
                (round_row[decrease_distance_to_c4_20s_cols] > 0.5) == ground_truth_tick_push_label
            )

            # need to filter only to labels for players that are alive
            alive_player_indices = [i for i, alive in enumerate(list(round_row[alive_cols])) if alive]
            correct_player_tick_label_5s = []
            correct_player_tick_label_10s = []
            correct_player_tick_label_20s = []
            for i in alive_player_indices:
                correct_player_tick_label_5s.append(correct_player_tick_label_5s_alive_and_dead[i])
                correct_player_tick_label_10s.append(correct_player_tick_label_10s_alive_and_dead[i])
                correct_player_tick_label_20s.append(correct_player_tick_label_20s_alive_and_dead[i])

            correct_tick_label_similarity = ground_truth_tick_push_label == predicted_tick_push_label
            correct_player_tick_label_similarity = [correct_tick_label_similarity for _ in correct_player_tick_label_5s]
            predicted_round_label_similarity_duplicated_per_player_tick = \
                [predicted_tick_push_label for _ in correct_player_tick_label_5s]

            if predicted_round_similarity_label < 0:
                if is_test_round:
                    test_result.num_not_labeled += len(correct_player_tick_label_5s)
                else:
                    train_result.num_not_labeled += len(correct_player_tick_label_5s)
                continue

            if is_test_round:
                result_to_update = test_result
            else:
                result_to_update = train_result
            result_to_update.correct_player_tick_label_5s += correct_player_tick_label_5s
            result_to_update.correct_player_tick_label_10s += correct_player_tick_label_10s
            result_to_update.correct_player_tick_label_20s += correct_player_tick_label_20s
            result_to_update.correct_player_tick_label_similarity += correct_player_tick_label_similarity
            result_to_update.predicted_round_label_similarity_duplicated_per_player_tick += \
                predicted_round_label_similarity_duplicated_per_player_tick
            pbar.update(1)

    train_5s_accuracy = sum(train_result.correct_player_tick_label_5s) / len(train_result.correct_player_tick_label_5s)
    train_10s_accuracy = sum(train_result.correct_player_tick_label_10s) / len(train_result.correct_player_tick_label_10s)
    train_20s_accuracy = sum(train_result.correct_player_tick_label_20s) / len(train_result.correct_player_tick_label_20s)
    train_similarity_accuracy = sum(train_result.correct_player_tick_label_similarity) / len(train_result.correct_player_tick_label_similarity)
    print(f"train 5s accuracy {train_5s_accuracy:.2f} train 10s accuracy {train_10s_accuracy:.2f} "
          f"train 20s accuracy {train_20s_accuracy:.2f} train similarity accuracy {train_similarity_accuracy:.2f}")

    test_5s_accuracy = sum(test_result.correct_player_tick_label_5s) / len(test_result.correct_player_tick_label_5s)
    test_10s_accuracy = sum(test_result.correct_player_tick_label_10s) / len(test_result.correct_player_tick_label_10s)
    test_20s_accuracy = sum(test_result.correct_player_tick_label_20s) / len(test_result.correct_player_tick_label_20s)
    test_similarity_accuracy = sum(test_result.correct_player_tick_label_similarity) / len(test_result.correct_player_tick_label_similarity)
    print(f"test 5s accuracy {test_5s_accuracy:.2f} test 10s accuracy {test_10s_accuracy:.2f} "
          f"test 20s accuracy {test_20s_accuracy:.2f} test similarity accuracy {test_similarity_accuracy:.2f}")


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
