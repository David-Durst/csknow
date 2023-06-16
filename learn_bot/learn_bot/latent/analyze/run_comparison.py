import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch

from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import max_enemies, round_id_column
from learn_bot.latent.place_area.column_names import place_area_input_column_types, delta_pos_output_column_types, \
    delta_pos_grid_num_cells
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.latent.vis.off_policy_inference import off_policy_inference
from learn_bot.latent.vis.vis_two import vis_two, PredictedToGroundTruthDict, PredictedToGroundTruthRoundData
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.latent.train import manual_latent_team_hdf5_data_path, rollout_latent_team_hdf5_data_path, \
    checkpoints_path, TrainResult, human_latent_team_hdf5_data_path, latent_id_cols
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd, HDF5Wrapper
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR
from learn_bot.latent.analyze.comparison_column_names import *

similarity_plots_path = Path(__file__).parent / 'similarity_plots'
similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'pathSimilarity.hdf5'
bot_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'manual_outputs' / 'botTrajectorySimilarity.hdf5'

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

bot_good_rounds = [4]
#bot_good_rounds = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
#                   57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 78, 80, 82, 85, 87, 89, 91, 93, 95, 97, 99, 102, 104, 106, 108,
#                   110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148,
#                   150, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190,
#                   192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230,
#                   232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270,
#                   272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310,
#                   312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350,
#                   352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 391,
#                   393, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432,
#                   434, 436, 438, 440, 442, 444, 446, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474,
#                   476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508, 510, 512, 514,
#                   516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 537, 539]

human_good_rounds = [512, 1, 4, 517, 520, 10, 15, 529, 534, 535, 25, 26, 27, 28, 541, 542, 544, 549, 38, 550, 552, 41, 42, 46, 558, 49, 566,
                     567, 56, 571, 61, 64, 65, 577, 67, 581, 71, 583, 584, 586, 587, 589, 592, 85, 88, 90, 91, 603, 96, 609, 610, 99, 101,
                     613, 615, 110, 111, 113, 626, 116, 629, 118, 122, 123, 124, 638, 127, 129, 641, 131, 133, 134, 135, 136, 137, 647, 139,
                     140, 652, 655, 144, 656, 148, 153, 666, 158, 159, 670, 162, 165, 166, 167, 678, 176, 691, 181, 182, 185, 699, 190, 706,
                     709, 199, 205, 207, 210, 217, 227, 234, 236, 237, 239, 242, 253, 255, 258, 261, 264, 269, 270, 276, 278, 280, 281, 299,
                     302, 303, 306, 308, 313, 321, 324, 326, 331, 335, 337, 345, 346, 347, 349, 355, 358, 365, 373, 375, 377, 380, 384, 394,
                     398, 408, 412, 422, 424, 427, 429, 431, 435, 439, 441, 442, 443, 451, 453, 456, 458, 461, 468, 479, 481, 484, 485, 488,
                     500, 507, 508]

#predicted_data_path = rollout_latent_team_hdf5_data_path
predicted_data_path = manual_latent_team_hdf5_data_path
#predicted_data_path = human_latent_team_hdf5_data_path
#ground_truth_data_path = rollout_latent_team_hdf5_data_path
ground_truth_data_path = manual_latent_team_hdf5_data_path
#ground_truth_data_path = human_latent_team_hdf5_data_path
limit_to_bot_good = True
limit_to_human_good = False
metric_cost_file_name = "bot_distribution"


def compare_trajectories():
    os.makedirs(similarity_plots_path, exist_ok=True)
    similarity_df = load_hdf5_to_pd(bot_similarity_hdf5_data_path)
    similarity_df = similarity_df[similarity_df[dtw_cost_col] != 0.]
    similarity_match_index_df = load_hdf5_to_pd(bot_similarity_hdf5_data_path, root_key='extra')

    if limit_to_bot_good:
        predicted_hdf5_wrapper = HDF5Wrapper(predicted_data_path, latent_id_cols)
        predicted_hdf5_wrapper.limit(predicted_hdf5_wrapper.id_df[round_id_column].isin(bot_good_rounds))
        predicted_df = load_hdf5_to_pd(predicted_data_path, rows_to_get=list(predicted_hdf5_wrapper.id_df['id'])).copy()
    elif limit_to_human_good:
        predicted_hdf5_wrapper = HDF5Wrapper(predicted_data_path, latent_id_cols)
        predicted_hdf5_wrapper.limit(predicted_hdf5_wrapper.id_df[round_id_column].isin(human_good_rounds))
        predicted_df = load_hdf5_to_pd(predicted_data_path, rows_to_get=list(predicted_hdf5_wrapper.id_df['id'])).copy()
    else:
        predicted_df = load_hdf5_to_pd(predicted_data_path).copy()
    predicted_result = load_model_file(predicted_df, "delta_pos_checkpoint.pt")

    predicted_to_ground_truth_dict: PredictedToGroundTruthDict = {}
    ground_truth_indices_ranges: List[range] = []
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000
    #print(similarity_df.loc[:, [predicted_round_id_col, best_fit_ground_truth_round_id_col, metric_type_col,
    #                            dtw_cost_col, delta_distance_col, delta_time_col]])

    # plot cost, distance, and time by metric type
    metric_types = similarity_df[metric_type_col].unique().tolist()
    metric_types_similarity_df = similarity_df.loc[:, [metric_type_col, dtw_cost_col, delta_distance_col, delta_time_col]]

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    axs = fig.subplots(len(metric_types), 3, squeeze=False)
    for i, metric_type in enumerate(metric_types):
        metric_type_str = metric_type.decode('utf-8')
        metric_type_similarity_df = metric_types_similarity_df[(similarity_df[metric_type_col] == metric_type)]
        metric_type_similarity_df.hist(dtw_cost_col, ax=axs[i, 0])
        axs[i, 0].set_title(metric_type_str + " DTW Cost")
        metric_type_similarity_df.hist(delta_distance_col, ax=axs[i, 1])
        axs[i, 1].set_title(metric_type_str + " Delta Distance")
        metric_type_similarity_df.hist(delta_time_col, ax=axs[i, 2])
        axs[i, 2].set_title(metric_type_str + " Delta Time")
    plt.savefig(similarity_plots_path / (metric_cost_file_name + '.png'))

    # multiple predicted rounds may match to same ground truth round, don't save them multiple times
    ground_truth_indices_ranges_set: set = set()
    for idx, row in similarity_df.iterrows():
        ground_truth_trace_range = range(row[best_fit_ground_truth_start_trace_index_col],
                                         row[best_fit_ground_truth_end_trace_index_col] + 1)
        if ground_truth_trace_range not in ground_truth_indices_ranges_set:
            ground_truth_indices_ranges.append(ground_truth_trace_range)
            ground_truth_indices_ranges_set.add(ground_truth_trace_range)

        metric_type = row[metric_type_col].decode('utf-8')
        similarity_match_df = similarity_match_index_df.iloc[row[start_dtw_matched_indices_col]:
                                                             row[start_dtw_matched_indices_col] + row[length_dtw_matched_inidices_col]]
        agent_mapping_str = row[agent_mapping_col].decode('utf-8')
        agent_mapping = {}
        for agent_pair in agent_mapping_str.split(','):
            agents = [int(agent) for agent in agent_pair.split('_')]
            agent_mapping[int(agents[0])] = int(agents[1])
        if row[predicted_round_id_col] not in predicted_to_ground_truth_dict:
            predicted_to_ground_truth_dict[row[predicted_round_id_col]] = {}
        predicted_to_ground_truth_dict[row[predicted_round_id_col]][metric_type] = \
            PredictedToGroundTruthRoundData(row[predicted_round_id_col], row[best_fit_ground_truth_round_id_col],
                                     row, similarity_match_df, agent_mapping)
        #similarity_match_name = f"{row[predicted_name_col].decode('utf-8')}_{metric_type}_vs_" \
        #                        f"{row[best_fit_ground_truth_name_col].decode('utf-8')}"
        #similarity_match_df.plot(first_matched_index_col, second_matched_index_col, title=similarity_match_name)
        #plt.savefig(similarity_plots_path / (similarity_match_name + '.png'))

    ground_truth_indices_ranges = sorted(ground_truth_indices_ranges, key=lambda r: r.start)
    ground_truth_indices = [i for r in ground_truth_indices_ranges for i in r]

    ground_truth_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=ground_truth_indices).copy()
    ground_truth_result = load_model_file(ground_truth_df, "delta_pos_checkpoint.pt")

    predicted_pred_pf = off_policy_inference(predicted_result.train_dataset, predicted_result.model,
                                           predicted_result.column_transformers)
    ground_truth_pred_pf = off_policy_inference(ground_truth_result.train_dataset, ground_truth_result.model,
                                          ground_truth_result.column_transformers)
    vis_two(predicted_df, predicted_pred_pf, ground_truth_df, ground_truth_pred_pf, predicted_to_ground_truth_dict)


if __name__ == "__main__":
    compare_trajectories()
