import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from learn_bot.latent.train import manual_latent_team_hdf5_data_path, rollout_latent_team_hdf5_data_path
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

similarity_plots_path = Path(__file__).parent / 'similarity_plots'
similiarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'pathSimilarity.hdf5'
predicted_name_col = 'predicted name'
best_fit_ground_truth_name_col = 'best fit ground truth name'
predicted_round_id_col = 'predicted round id'
best_fit_ground_truth_round_id_col = 'best fit ground truth round id'
predicted_start_trace_index_col = 'predicted start trace index'
predicted_end_trace_index_col = 'predicted end trace index'
best_fit_ground_truth_start_trace_index_col = 'best fit ground truth start trace index'
best_fit_ground_truth_end_trace_index_col = 'best fit ground truth end trace index'
dtw_cost_col = 'dtw cost'
delta_time_col = 'delta time'
delta_distance_col = 'delta distance'
start_dtw_matched_indices_col = 'start dtw matched indices'
length_dtw_matched_inidices_col = 'length dtw matched indices'
first_matched_index_col = 'first matched index'
second_matched_index_col = 'second matched index'


def compare_trajectories():
    os.makedirs(similarity_plots_path, exist_ok=True)
    similarity_df = load_hdf5_to_pd(similiarity_hdf5_data_path)
    similarity_match_index_df = load_hdf5_to_pd(similiarity_hdf5_data_path, root_key='extra')
    rollout_df = load_hdf5_to_pd(rollout_latent_team_hdf5_data_path)

    manual_indices_ranges: List[range] = []
    for _, row in similarity_df.iterrows():
        manual_indices_ranges.append(range(row[best_fit_ground_truth_start_trace_index_col],
                                           row[best_fit_ground_truth_end_trace_index_col] + 1))

        similarity_match_name = row[predicted_name_col].decode('utf-8') + "_vs_" + row[best_fit_ground_truth_name_col].decode('utf-8')
        similarity_match_df = similarity_match_index_df.iloc[row[start_dtw_matched_indices_col]:
                                                             row[start_dtw_matched_indices_col] + row[length_dtw_matched_inidices_col]]
        similarity_axes = \
            similarity_match_df.plot(first_matched_index_col, second_matched_index_col, title=similarity_match_name)
        plt.savefig(similarity_plots_path / (similarity_match_name + '.png'))

    manual_indices_ranges = sorted(manual_indices_ranges, key=lambda r: r.start)
    manual_indices = [i for r in manual_indices_ranges for i in r]

    manual_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=manual_indices)


if __name__ == "__main__":
    compare_trajectories()
