import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch

from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.place_area.column_names import place_area_input_column_types, delta_pos_output_column_types, \
    delta_pos_grid_num_cells
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.latent.vis.off_policy_inference import off_policy_inference
from learn_bot.latent.vis.vis_two import vis_two, RolloutToManualDict, RolloutToManualRoundData
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.latent.train import manual_latent_team_hdf5_data_path, rollout_latent_team_hdf5_data_path, \
    checkpoints_path, TrainResult
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR
from learn_bot.latent.analyze.comparison_column_names import *

similarity_plots_path = Path(__file__).parent / 'similarity_plots'
similiarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'pathSimilarity.hdf5'

def load_model_file(all_data_df: pd.DataFrame, model_file_name: str) -> TrainResult:
    cur_checkpoints_path = checkpoints_path
    if len(sys.argv) > 1:
        cur_checkpoints_path = cur_checkpoints_path / sys.argv[1]
    model_file = torch.load(cur_checkpoints_path / model_file_name)

    make_index_column(all_data_df)

    all_data = LatentDataset(all_data_df, model_file['column_transformers'])

    column_transformers = IOColumnTransformers(place_area_input_column_types, delta_pos_output_column_types, all_data_df)

    model = TransformerNestedHiddenLatentModel(model_file['column_transformers'], 2 * max_enemies, delta_pos_grid_num_cells, 2, 4)
    model.load_state_dict(model_file['model_state_dict'])
    model.to(CUDA_DEVICE_STR)

    return TrainResult(all_data, all_data, all_data_df, all_data_df, column_transformers, model)


def compare_trajectories():
    os.makedirs(similarity_plots_path, exist_ok=True)
    similarity_df = load_hdf5_to_pd(similiarity_hdf5_data_path)
    similarity_match_index_df = load_hdf5_to_pd(similiarity_hdf5_data_path, root_key='extra')
    rollout_df = load_hdf5_to_pd(rollout_latent_team_hdf5_data_path).copy()
    rollout_result = load_model_file(rollout_df, "delta_pos_checkpoint.pt")

    rollout_to_manual_dict: RolloutToManualDict = {}
    manual_indices_ranges: List[range] = []
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000
    print(similarity_df.loc[:, [predicted_round_id_col, best_fit_ground_truth_round_id_col, metric_type_col,
                                dtw_cost_col, delta_distance_col, delta_time_col]])
    # multiple predicted rounds may match to same ground truth round, don't save them multiple times
    manual_indices_ranges_set: set = set()
    for idx, row in similarity_df.iterrows():
        ground_truth_trace_range = range(row[best_fit_ground_truth_start_trace_index_col],
                                         row[best_fit_ground_truth_end_trace_index_col] + 1)
        if ground_truth_trace_range not in manual_indices_ranges_set:
            manual_indices_ranges.append(ground_truth_trace_range)
            manual_indices_ranges_set.add(ground_truth_trace_range)

        metric_type = row[metric_type_col].decode('utf-8')
        similarity_match_name = f"{row[predicted_name_col].decode('utf-8')}_{metric_type}_vs_" \
                                f"{row[best_fit_ground_truth_name_col].decode('utf-8')}"
        similarity_match_df = similarity_match_index_df.iloc[row[start_dtw_matched_indices_col]:
                                                             row[start_dtw_matched_indices_col] + row[length_dtw_matched_inidices_col]]
        agent_mapping_str = row[agent_mapping_col].decode('utf-8')
        agent_mapping = {}
        for agent_pair in agent_mapping_str.split(','):
            agents = [int(agent) for agent in agent_pair.split('_')]
            agent_mapping[int(agents[0])] = int(agents[1])
        if row[predicted_round_id_col] not in rollout_to_manual_dict:
            rollout_to_manual_dict[row[predicted_round_id_col]] = {}
        rollout_to_manual_dict[row[predicted_round_id_col]][metric_type] = \
            RolloutToManualRoundData(row[predicted_round_id_col], row[best_fit_ground_truth_round_id_col],
                                     row, similarity_match_df, agent_mapping)
        similarity_match_df.plot(first_matched_index_col, second_matched_index_col, title=similarity_match_name)
        plt.savefig(similarity_plots_path / (similarity_match_name + '.png'))

    manual_indices_ranges = sorted(manual_indices_ranges, key=lambda r: r.start)
    manual_indices = [i for r in manual_indices_ranges for i in r]

    manual_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=manual_indices).copy()
    manual_result = load_model_file(manual_df, "delta_pos_checkpoint.pt")

    rollout_pred_pf = off_policy_inference(rollout_result.train_dataset, rollout_result.model,
                                           rollout_result.column_transformers)
    manual_pred_pf = off_policy_inference(manual_result.train_dataset, manual_result.model,
                                          manual_result.column_transformers)
    vis_two(rollout_df, rollout_pred_pf, manual_df, manual_pred_pf, rollout_to_manual_dict)


if __name__ == "__main__":
    compare_trajectories()
