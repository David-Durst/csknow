import dataclasses
import os
import sys

import time

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import ComparisonConfig, \
    plot_trajectory_comparison_histograms, build_predicted_to_ground_truth_dict
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.vis.vis_two import vis_two
from learn_bot.latent.place_area.load_data import LoadDataOptions, LoadDataResult
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.latent.analyze.comparison_column_names import *

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

manual_load_data_option = LoadDataOptions(
    use_manual_data=True,
    use_rollout_data=False,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_all_human_data=False,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=False
)

hand_crafted_bot_vs_hand_crafted_bot_config = ComparisonConfig(
    hand_crafted_bot_vs_hand_crafted_bot_similarity_hdf5_data_path,
    manual_load_data_option,
    manual_load_data_option,
    True,
    False,
    "hand_crafted_bot_vs_hand_crafted_bot_distribution",
    "Hand-Crafted Bot vs Hand-Crafted Bot Distribution"
)

rollout_load_data_option = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=True,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_all_human_data=False,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=False
)

learned_time_bot_vs_hand_crafted_bot_config = ComparisonConfig(
    time_vs_hand_crafted_bot_similarity_hdf5_data_path,
    rollout_load_data_option,
    manual_load_data_option,
    False,
    False,
    "learned_time_bot_vs_hand_crafted_bot_distribution",
    "Learned Time Bot vs Hand-Crafted Bot Distribution"
)

learned_no_time_bot_vs_hand_crafted_bot_config = ComparisonConfig(
    no_time_vs_hand_crafted_bot_similarity_hdf5_data_path,
    rollout_load_data_option,
    manual_load_data_option,
    False,
    False,
    "learned_no_time_no_weight_decay_bot_vs_hand_crafted_bot_distribution",
    "Learned No Time No Weight Decay Bot vs Hand-Crafted Bot Distribution"
)

small_human_load_data_option = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=False,
    use_synthetic_data=False,
    use_small_human_data=True,
    use_all_human_data=False,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=False
)

human_vs_human_config = ComparisonConfig(
    human_vs_human_similarity_hdf5_data_path,
    small_human_load_data_option,
    small_human_load_data_option,
    False,
    True,
    "human_vs_human_distribution",
    "Human vs Human Distribution"
)

all_human_load_data_option = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=False,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_all_human_data=True,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=False
)

all_human_vs_all_human_config = ComparisonConfig(
    all_human_vs_all_human_similarity_hdf5_data_path,
    all_human_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "all_human_vs_all_human_distribution",
    "All Human vs All Human Distribution"
)

all_human_vs_small_human_config = ComparisonConfig(
    all_human_vs_small_human_similarity_hdf5_data_path,
    all_human_load_data_option,
    small_human_load_data_option,
    False,
    False,
    "all_human_vs_small_human_distribution",
    "All Human vs Small Human Distribution"
)

rollout_vs_all_human_config = ComparisonConfig(
    rollout_vs_all_human_similarity_hdf5_data_path,
    rollout_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "rollout_vs_all_human_distribution",
    "Rollout vs All Human Distribution"
)

rollout_learned_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_9_10_23_durst_learned")
rollout_learned_vs_all_human_config = ComparisonConfig(
    rollout_learned_vs_all_human_similarity_hdf5_data_path,
    rollout_learned_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "rollout_learned_vs_all_human_distribution",
    "Rollout Learned vs All Human Distribution"
)

rollout_handcrafted_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_9_10_23_durst_handcrafted")
rollout_handcrafted_vs_all_human_config = ComparisonConfig(
    rollout_handcrafted_vs_all_human_similarity_hdf5_data_path,
    rollout_handcrafted_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "rollout_handcrafted_vs_all_human_distribution",
    "Rollout Hand-Crafted vs All Human Distribution"
)

rollout_default_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                           custom_rollout_extension= "_9_10_23_default")
rollout_default_vs_all_human_config = ComparisonConfig(
    rollout_default_vs_all_human_similarity_hdf5_data_path,
    rollout_default_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "rollout_default_vs_all_human_distribution",
    "Rollout Default vs All Human Distribution"
)

just_plot_summaries = True

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
    elif config_case == 5:
        config = all_human_vs_small_human_config
    elif config_case == 6:
        config = rollout_vs_all_human_config
    elif config_case == 7:
        config = rollout_learned_vs_all_human_config
    elif config_case == 8:
        config = rollout_handcrafted_vs_all_human_config
    elif config_case == 9:
        config = rollout_default_vs_all_human_config

    os.makedirs(similarity_plots_path, exist_ok=True)
    similarity_df = load_hdf5_to_pd(config.similarity_data_path)
    similarity_df = similarity_df[similarity_df[dtw_cost_col] != 0.]
    # remove this if later when updated all comparisons
    if config_case == 5:
        similarity_df = similarity_df[(similarity_df[predicted_round_number_col] != similarity_df[best_fit_ground_truth_round_number_col]) |
                                      (abs(similarity_df[predicted_first_game_tick_number_col] - similarity_df[best_fit_ground_truth_first_game_tick_number_col]) > 11)]
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
    predicted_data = LoadDataResult(config.predicted_load_data_options)
    if config.limit_predicted_df_to_bot_good:
        predicted_data.limit([lambda df: df[round_id_column].isin(bot_good_rounds)])
        #predicted_hdf5_wrapper = HDF5Wrapper(config.predicted_data_path, latent_id_cols)
        #predicted_hdf5_wrapper.limit(predicted_hdf5_wrapper.id_df[round_id_column].isin(bot_good_rounds))
        #predicted_df = load_hdf5_to_pd(config.predicted_data_path)
        #predicted_df = predicted_df.iloc[predicted_hdf5_wrapper.id_df['id'], :]
    elif config.limit_predicted_df_to_human_good:
        predicted_data.limit([lambda df: df[round_id_column].isin(small_human_good_rounds)])
        #predicted_hdf5_wrapper = HDF5Wrapper(config.predicted_data_path, latent_id_cols)
        #predicted_hdf5_wrapper.limit(predicted_hdf5_wrapper.id_df[round_id_column].isin(human_good_rounds))
        #predicted_df = load_hdf5_to_pd(config.predicted_data_path)
        #predicted_df = predicted_df.iloc[predicted_hdf5_wrapper.id_df['id'], :]
    #else:
    #    predicted_df = load_hdf5_to_pd(config.predicted_data_path)
    predicted_model = load_model_file(predicted_data)
    end_predicted_load_time = time.perf_counter()
    print(f"predicted load time {end_predicted_load_time - start_predicted_load_time: 0.4f}")

    start_ground_truth_load_time = time.perf_counter()

    ground_truth_data = LoadDataResult(config.ground_truth_load_data_options)
    ground_truth_model = load_model_file(ground_truth_data)
    end_ground_truth_load_time = time.perf_counter()
    print(f"ground truth load time {end_ground_truth_load_time - start_ground_truth_load_time: 0.4f}")

    vis_two(predicted_model, ground_truth_model, predicted_to_ground_truth_dict, similarity_match_index_df)


if __name__ == "__main__":
    compare_trajectories()
