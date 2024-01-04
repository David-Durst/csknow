import dataclasses

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.place_area.column_names import get_similarity_column
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.place_area.push_save_label import PushSaveRoundLabels
from learn_bot.latent.train_paths import default_save_push_round_labels_path
from learn_bot.latent.vis import run_vis_checkpoint
from learn_bot.libs.multi_hdf5_wrapper import absolute_to_relative_train_test_key

fig_length = 6


def leave_one_out_similarity_analysis(load_data_result: LoadDataResult):
    push_save_round_labels: PushSaveRoundLabels = PushSaveRoundLabels()
    push_save_round_labels.load(default_save_push_round_labels_path)

    # labels are on first hdf5 file
    hdf5_wrapper = load_data_result.multi_hdf5_wrapper.hdf5_wrappers[0]
    hdf5_key = absolute_to_relative_train_test_key(hdf5_wrapper.hdf5_path)
    test_group_ids = load_data_result.multi_hdf5_wrapper.test_group_ids[hdf5_key]
    # relying on 28th file to come first, which always has been true
    assert '_28.hdf5' in str(hdf5_key)

    # compute accuracy for all round ids, recording if in test or train set
    num_train_not_labeled = 0
    train_abs_similarity_errors = []
    num_test_not_labeled = 0
    test_abs_similarity_errors = []
    round_ids_and_similarity_df = \
        hdf5_wrapper.id_df.groupby(round_id_column, as_index=False)[get_similarity_column(0)].first()
    for _, round_id_and_similarity_row in round_ids_and_similarity_df.iterrows():
        round_id = round_id_and_similarity_row[round_id_column]
        is_test_round = round_id in test_group_ids
        similarity_label = round_id_and_similarity_row[get_similarity_column(0)]
        similarity_error = abs(similarity_label - push_save_round_labels.round_id_to_data[round_id].to_float_label())
        not_labeled = similarity_label < 0.
        if is_test_round:
            if not_labeled:
                num_test_not_labeled += 1
            else:
                test_abs_similarity_errors.append(similarity_error)
        else:
            if not_labeled:
                num_train_not_labeled += 1
            else:
                train_abs_similarity_errors.append(similarity_error)

    # plot two histograms
    train_abs_similarity_errors_np = np.array(train_abs_similarity_errors)
    train_abs_similarity_errors_series = pd.Series(train_abs_similarity_errors)
    test_abs_similarity_errors_np = np.array(test_abs_similarity_errors)
    test_abs_similarity_errors_series = pd.Series(test_abs_similarity_errors)
    bins = [i * 0.1 for i in range(11)]
    plt.figure(figsize=(fig_length, fig_length), constrained_layout=True)
    ax = plt.gca()
    ax.set_title('Similarity Errors')
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    train_weights = np.ones(len(train_abs_similarity_errors)) / len(train_abs_similarity_errors)
    test_weights = np.ones(len(test_abs_similarity_errors)) / len(test_abs_similarity_errors)
    ax.hist([train_abs_similarity_errors_np, test_abs_similarity_errors_np], bins,
            histtype='bar', label=['Train', 'Test'], weights=[train_weights, test_weights])
    train_description = 'Train\n' + train_abs_similarity_errors_series.describe().to_string() + \
                        f'\n not labeled {num_train_not_labeled} '
    ax.text(0.2, 0.2, train_description, family='monospace')
    test_description = 'Test\n' + test_abs_similarity_errors_series.describe().to_string() + \
                        f'\n not labeled {num_test_not_labeled} '
    ax.text(0.65, 0.2, test_description, family='monospace')
    ax.legend()
    plt.savefig(similarity_plots_path / 'similarity_errors.png')


if __name__ == "__main__":
    load_data_options = dataclasses.replace(run_vis_checkpoint.load_data_options, similarity_analysis=True)
    load_data_result = LoadDataResult(load_data_options=load_data_options)
    leave_one_out_similarity_analysis(load_data_result)
