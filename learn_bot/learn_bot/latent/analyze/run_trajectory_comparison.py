import gc
import os
import sys
from pathlib import Path
from typing import List

from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import time

from matplotlib.ticker import PercentFormatter

from learn_bot.latent.analyze.process_trajectory_comparison import ComparisonConfig, \
    plot_trajectory_comparison_histograms, build_predicted_to_ground_truth_dict
from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import max_enemies, round_id_column
from learn_bot.latent.place_area.column_names import place_area_input_column_types, delta_pos_output_column_types, \
    delta_pos_grid_num_cells
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.latent.vis.off_policy_inference import off_policy_inference
from learn_bot.latent.vis.vis_two import vis_two, PredictedToGroundTruthDict, PredictedToGroundTruthRoundData
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.latent.train import checkpoints_path, TrainResult, latent_id_cols
from learn_bot.latent.place_area.load_data import human_latent_team_hdf5_data_path, manual_latent_team_hdf5_data_path, \
    rollout_latent_team_hdf5_data_path, all_train_latent_team_hdf5_dir_path
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR
from learn_bot.latent.analyze.comparison_column_names import *

similarity_plots_path = Path(__file__).parent / 'similarity_plots'
hand_crafted_bot_vs_hand_crafted_bot_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'manual_outputs' / 'botTrajectorySimilarity.hdf5'
time_vs_hand_crafted_bot_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'learnedTimeBotTrajectorySimilarity.hdf5'
no_time_vs_hand_crafted_bot_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'learnedNoTimeNoWeightDecayBotTrajectorySimilarity.hdf5'
human_vs_human_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'humanTrajectorySimilarity.hdf5'
all_human_vs_all_human_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs' / 'humanTrajectorySimilarity.hdf5'

def load_model_file(all_data_df: pd.DataFrame, model_file_name: str) -> TrainResult:
    cur_checkpoints_path = checkpoints_path
    if len(sys.argv) > 2:
        cur_checkpoints_path = cur_checkpoints_path / sys.argv[2]
    model_file = torch.load(cur_checkpoints_path / model_file_name)

    make_index_column(all_data_df)

    all_data = LatentDataset(all_data_df, model_file['column_transformers'])

    column_transformers = IOColumnTransformers(place_area_input_column_types, delta_pos_output_column_types, all_data_df)

    model = TransformerNestedHiddenLatentModel(model_file['column_transformers'], 2 * max_enemies, delta_pos_grid_num_cells, 2, 4)
    model.load_state_dict(model_file['model_state_dict'])
    model.to(CUDA_DEVICE_STR)

    return TrainResult(all_data, all_data, all_data_df, all_data_df, column_transformers, model)

bot_good_rounds = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
                   57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 78, 80, 82, 85, 87, 89, 91, 93, 95, 97, 99, 102, 104, 106, 108,
                   110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148,
                   150, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190,
                   192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230,
                   232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270,
                   272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310,
                   312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350,
                   352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 391,
                   393, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432,
                   434, 436, 438, 440, 442, 444, 446, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474,
                   476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508, 510, 512, 514,
                   516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 537, 539]

human_good_rounds = [512, 1, 4, 517, 520, 10, 15, 529, 534, 535, 25, 26, 27, 28, 541, 542, 544, 549, 38, 550, 552, 41, 42, 46, 558, 49, 566,
                     567, 56, 571, 61, 64, 65, 577, 67, 581, 71, 583, 584, 586, 587, 589, 592, 85, 88, 90, 91, 603, 96, 609, 610, 99, 101,
                     613, 615, 110, 111, 113, 626, 116, 629, 118, 122, 123, 124, 638, 127, 129, 641, 131, 133, 134, 135, 136, 137, 647, 139,
                     140, 652, 655, 144, 656, 148, 153, 666, 158, 159, 670, 162, 165, 166, 167, 678, 176, 691, 181, 182, 185, 699, 190, 706,
                     709, 199, 205, 207, 210, 217, 227, 234, 236, 237, 239, 242, 253, 255, 258, 261, 264, 269, 270, 276, 278, 280, 281, 299,
                     302, 303, 306, 308, 313, 321, 324, 326, 331, 335, 337, 345, 346, 347, 349, 355, 358, 365, 373, 375, 377, 380, 384, 394,
                     398, 408, 412, 422, 424, 427, 429, 431, 435, 439, 441, 442, 443, 451, 453, 456, 458, 461, 468, 479, 481, 484, 485, 488,
                     500, 507, 508]

#predicted_data_path = rollout_latent_team_hdf5_data_path
#predicted_data_path = manual_latent_team_hdf5_data_path
#predicted_data_path = human_latent_team_hdf5_data_path
#ground_truth_data_path = rollout_latent_team_hdf5_data_path
#ground_truth_data_path = manual_latent_team_hdf5_data_path
#ground_truth_data_path = human_latent_team_hdf5_data_path
#limit_to_bot_good = False
#limit_to_human_good = False
#metric_cost_file_name = "hand_crafted_bot_vs_hand_crafted_bot_distribution"
#metric_cost_title = "Hand-Crafted Bot vs Hand-Crafted Bot Distribution"
#metric_cost_file_name = "learned_time_bot_vs_hand_crafted_bot_distribution"
#metric_cost_title = "Learned Time Bot vs Hand-Crafted Bot Distribution"
#metric_cost_file_name = "learned_no_time_no_weight_decay_bot_vs_hand_crafted_bot_distribution"
#metric_cost_title = "Learned No Time No Weight Decay Bot vs Hand-Crafted Bot Distribution"
#metric_cost_file_name = "human_vs_human_distribution"
#metric_cost_title = "Human vs Human Distribution"

hand_crafted_bot_vs_hand_crafted_bot_config = ComparisonConfig(
    hand_crafted_bot_vs_hand_crafted_bot_similarity_hdf5_data_path,
    manual_latent_team_hdf5_data_path,
    manual_latent_team_hdf5_data_path,
    True,
    False,
    "hand_crafted_bot_vs_hand_crafted_bot_distribution",
    "Hand-Crafted Bot vs Hand-Crafted Bot Distribution"
)


learned_time_bot_vs_hand_crafted_bot_config = ComparisonConfig(
    time_vs_hand_crafted_bot_similarity_hdf5_data_path,
    rollout_latent_team_hdf5_data_path,
    manual_latent_team_hdf5_data_path,
    False,
    False,
    "learned_time_bot_vs_hand_crafted_bot_distribution",
    "Learned Time Bot vs Hand-Crafted Bot Distribution"
)

learned_no_time_bot_vs_hand_crafted_bot_config = ComparisonConfig(
    no_time_vs_hand_crafted_bot_similarity_hdf5_data_path,
    rollout_latent_team_hdf5_data_path,
    manual_latent_team_hdf5_data_path,
    False,
    False,
    "learned_no_time_no_weight_decay_bot_vs_hand_crafted_bot_distribution",
    "Learned No Time No Weight Decay Bot vs Hand-Crafted Bot Distribution"
)

human_vs_human_config = ComparisonConfig(
    human_vs_human_similarity_hdf5_data_path,
    human_latent_team_hdf5_data_path,
    human_latent_team_hdf5_data_path,
    False,
    True,
    "human_vs_human_distribution",
    "Human vs Human Distribution"
)

all_human_vs_all_human_config = ComparisonConfig(
    all_human_vs_all_human_similarity_hdf5_data_path,
    all_train_latent_team_hdf5_dir_path, # TODO: need to update this for multi hdf5 wrapper, then change these paths
    all_train_latent_team_hdf5_dir_path,
    False,
    False,
    "all_human_vs_all_human_distribution",
    "All Human vs All Human Distribution"
)


just_plot_summaries = False

def compare_trajectories():
    config_case = int(sys.argv[2])
    if config_case == 0:
        config = hand_crafted_bot_vs_hand_crafted_bot_config
    elif config_case == 1:
        config = learned_time_bot_vs_hand_crafted_bot_config
    elif config_case == 2:
        config = learned_no_time_bot_vs_hand_crafted_bot_config
    elif config_case == 3:
        config = human_vs_human_config
    elif config_case == 4:
        config = all_human_vs_all_human_config

    os.makedirs(similarity_plots_path, exist_ok=True)
    similarity_df = load_hdf5_to_pd(config.similarity_data_path)
    similarity_df = similarity_df[similarity_df[dtw_cost_col] != 0.]
    similarity_match_index_df = load_hdf5_to_pd(config.similarity_data_path, root_key='extra')

    start_similarity_plot_time = time.perf_counter()
    plot_trajectory_comparison_histograms(similarity_df, config, similarity_plots_path)
    end_similarity_plot_time = time.perf_counter()
    print(f"similarity plot time {end_similarity_plot_time - start_similarity_plot_time: 0.4f}")

    if just_plot_summaries:
        exit(0)

    # computing mapping between predict and ground truth
    # multiple predicted rounds may match to same ground truth round, don't save them multiple times
    start_predicted_to_ground_truth_time = time.perf_counter()
    predicted_to_ground_truth_dict = build_predicted_to_ground_truth_dict(similarity_df)
    end_predicted_to_ground_truth_time = time.perf_counter()
    print(f"predicted to ground truth time {end_predicted_to_ground_truth_time - start_predicted_to_ground_truth_time: 0.4f}")

    # load data
    start_predicted_load_time = time.perf_counter()
    if config.limit_predicted_df_to_bot_good:
        predicted_hdf5_wrapper = HDF5Wrapper(config.predicted_data_path, latent_id_cols)
        predicted_hdf5_wrapper.limit(predicted_hdf5_wrapper.id_df[round_id_column].isin(bot_good_rounds))
        predicted_df = load_hdf5_to_pd(config.predicted_data_path)
        predicted_df = predicted_df.iloc[predicted_hdf5_wrapper.id_df['id'], :]
    elif config.limit_predicted_df_to_human_good:
        predicted_hdf5_wrapper = HDF5Wrapper(config.predicted_data_path, latent_id_cols)
        predicted_hdf5_wrapper.limit(predicted_hdf5_wrapper.id_df[round_id_column].isin(human_good_rounds))
        #predicted_df = load_hdf5_to_pd(config.predicted_data_path)
        #predicted_df = predicted_df.iloc[predicted_hdf5_wrapper.id_df['id'], :]
    else:
        predicted_df = load_hdf5_to_pd(config.predicted_data_path)
    print(len(predicted_df))
    time.sleep(10)
    exit(0)
    predicted_result = load_model_file(predicted_df, "delta_pos_checkpoint.pt")
    end_predicted_load_time = time.perf_counter()
    print(f"predicted load time {end_predicted_load_time - start_predicted_load_time: 0.4f}")

    start_ground_truth_load_time = time.perf_counter()
    ground_truth_indices_ranges = sorted(ground_truth_indices_ranges, key=lambda r: r.start)
    ground_truth_indices = [i for r in ground_truth_indices_ranges for i in r]

    ground_truth_df = load_hdf5_to_pd(config.ground_truth_data_path)
    ground_truth_df = ground_truth_df.iloc[ground_truth_indices, :]
    ground_truth_result = load_model_file(ground_truth_df, "delta_pos_checkpoint.pt")
    end_ground_truth_load_time = time.perf_counter()
    print(f"ground truth load time {end_ground_truth_load_time - start_ground_truth_load_time: 0.4f}")

    vis_two(predicted_df, ground_truth_df, predicted_to_ground_truth_dict)


if __name__ == "__main__":
    compare_trajectories()
