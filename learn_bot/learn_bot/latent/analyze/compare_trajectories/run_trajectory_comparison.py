import dataclasses
import os
import sys

import time
from typing import Optional

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_from_comparison import \
    plot_trajectory_comparisons, TrajectoryPlots, concat_trajectory_plots_across_player_type
from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import ComparisonConfig, \
    plot_trajectory_comparison_histograms, build_predicted_to_ground_truth_dict
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.train_paths import train_test_split_file_name
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
    use_human_28_data=False,
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
    use_human_28_data=False,
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
    use_human_28_data=False,
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
    use_human_28_data=False,
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

human_28_load_data_option = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=False,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_human_28_data=True,
    use_all_human_data=False,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=False
)

all_human_vs_human_28_config = ComparisonConfig(
    all_human_vs_human_28_similarity_hdf5_data_path,
    all_human_load_data_option,
    human_28_load_data_option,
    False,
    False,
    "all_human_vs_human_28_distribution",
    "All Human vs Human 28 Distribution"
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

#rollout_learned_load_data_option = dataclasses.replace(rollout_load_data_option,
#                                                       custom_rollout_extension= "_10_03_23_learned_250ms_scheduled_sampling_300_rounds")
#custom_rollout_extension = "_9_20_23_learned_3s_300_rounds")
rollout_learned_no_mask_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_learned_no_mask_10_19_23_300_rounds")
rollout_learned_no_mask_vs_all_human_similarity_hdf5_data_path = \
    rollout_learned_vs_all_human_similarity_hdf5_data_path.parent / \
    str(rollout_learned_vs_all_human_similarity_hdf5_data_path.name).replace("learned", rollout_learned_no_mask_load_data_option.custom_rollout_extension[1:])
rollout_learned_no_mask_vs_all_human_config = ComparisonConfig(
    rollout_learned_no_mask_vs_all_human_similarity_hdf5_data_path,
    rollout_learned_no_mask_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "rollout_learned_no_mask_vs_all_human_distribution",
    "Rollout Learned No Mask vs All Human Distribution"
)

rollout_learned_teammate_mask_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_learned_teammate_mask_10_19_23_300_rounds")
rollout_learned_teammate_mask_vs_all_human_similarity_hdf5_data_path = \
    rollout_learned_vs_all_human_similarity_hdf5_data_path.parent / \
    str(rollout_learned_vs_all_human_similarity_hdf5_data_path.name).replace("learned", rollout_learned_teammate_mask_load_data_option.custom_rollout_extension[1:])
rollout_learned_teammate_mask_vs_all_human_config = ComparisonConfig(
    rollout_learned_teammate_mask_vs_all_human_similarity_hdf5_data_path,
    rollout_learned_teammate_mask_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "rollout_learned_teammate_mask_vs_all_human_distribution",
    "Rollout Teammate Mask Learned vs All Human Distribution"
)

rollout_learned_enemy_mask_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                                     custom_rollout_extension= "_learned_enemy_mask_10_19_23_300_rounds")
rollout_learned_enemy_mask_vs_all_human_similarity_hdf5_data_path = \
    rollout_learned_vs_all_human_similarity_hdf5_data_path.parent / \
    str(rollout_learned_vs_all_human_similarity_hdf5_data_path.name).replace("learned", rollout_learned_enemy_mask_load_data_option.custom_rollout_extension[1:])
rollout_learned_enemy_mask_vs_all_human_config = ComparisonConfig(
    rollout_learned_enemy_mask_vs_all_human_similarity_hdf5_data_path,
    rollout_learned_enemy_mask_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "rollout_learned_enemy_mask_vs_all_human_distribution",
    "Rollout Learned Enemy Mask vs All Human Distribution"
)

rollout_learned_everyone_mask_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                                  custom_rollout_extension= "_learned_everyone_mask_10_19_23_300_rounds")
rollout_learned_everyone_mask_vs_all_human_similarity_hdf5_data_path = \
    rollout_learned_vs_all_human_similarity_hdf5_data_path.parent / \
    str(rollout_learned_vs_all_human_similarity_hdf5_data_path.name).replace("learned", rollout_learned_everyone_mask_load_data_option.custom_rollout_extension[1:])
rollout_learned_everyone_mask_vs_all_human_config = ComparisonConfig(
    rollout_learned_everyone_mask_vs_all_human_similarity_hdf5_data_path,
    rollout_learned_everyone_mask_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "rollout_learned_everyone_mask_vs_all_human_distribution",
    "Rollout Learned Everyone Mask vs All Human Distribution"
)

#rollout_no_history_learned_load_data_option = dataclasses.replace(rollout_load_data_option,
#                                                       custom_rollout_extension="_9_20_23_learned_3s_300_rounds")
#rollout_no_history_learned_vs_all_human_similarity_hdf5_data_path = \
#    rollout_learned_vs_all_human_similarity_hdf5_data_path.parent / \
#    str(rollout_learned_vs_all_human_similarity_hdf5_data_path.name).replace("learned", rollout_no_history_learned_load_data_option.custom_rollout_extension[1:])
#rollout_no_history_learned_vs_all_human_config = ComparisonConfig(
#    rollout_no_history_learned_vs_all_human_similarity_hdf5_data_path,
#    rollout_no_history_learned_load_data_option,
#    all_human_load_data_option,
#    False,
#    False,
#    "rollout_no_history_learned_vs_all_human_distribution",
#    "Rollout No History Learned vs All Human Distribution"
#)

rollout_handcrafted_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_handcrafted_10_19_23_300_rounds")
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
                                                           custom_rollout_extension= "_default_10_19_23_300_rounds")
rollout_default_vs_all_human_config = ComparisonConfig(
    rollout_default_vs_all_human_similarity_hdf5_data_path,
    rollout_default_load_data_option,
    all_human_load_data_option,
    False,
    False,
    "rollout_default_vs_all_human_distribution",
    "Rollout Default vs All Human Distribution"
)

test_all_human_load_data_option = dataclasses.replace(all_human_load_data_option,
                                                      train_test_split_file_name=train_test_split_file_name)

rollout_all_human_vs_learned_no_mask_similarity_hdf5_data_path = \
    rollout_all_human_vs_learned_similarity_hdf5_data_path.parent / \
    'human_to_learned_no_mask_10_19_23_300_rounds_trajectorySimilarity.hdf5'
rollout_all_human_vs_learned_config = ComparisonConfig(
    rollout_all_human_vs_learned_no_mask_similarity_hdf5_data_path,
    test_all_human_load_data_option,
    rollout_learned_no_mask_load_data_option,
    False,
    False,
    "rollout_all_human_vs_learned_distribution",
    "Rollout All Human vs Learned Distribution",
    True
)

rollout_all_human_vs_handcrafted_config = ComparisonConfig(
    rollout_all_human_vs_handcrafted_similarity_hdf5_data_path,
    test_all_human_load_data_option,
    rollout_handcrafted_load_data_option,
    False,
    False,
    "rollout_all_human_vs_handcrafted_distribution",
    "Rollout All Human vs Hand-Crafted Distribution",
    True
)

rollout_all_human_vs_default_config = ComparisonConfig(
    rollout_all_human_vs_default_similarity_hdf5_data_path,
    test_all_human_load_data_option,
    rollout_default_load_data_option,
    False,
    False,
    "rollout_all_human_vs_default_distribution",
    "Rollout All Human vs Default Distribution",
    True
)

just_plot_summaries = False
plot_trajectories = False


def compare_trajectories(config_case: int) -> Optional[TrajectoryPlots]:
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
        config = all_human_vs_human_28_config
    elif config_case == 6:
        config = rollout_vs_all_human_config
    elif config_case == 7:
        config = rollout_learned_no_mask_vs_all_human_config
    elif config_case == 8:
        config = rollout_learned_teammate_mask_vs_all_human_config
    elif config_case == 9:
        config = rollout_learned_enemy_mask_vs_all_human_config
    elif config_case == 10:
        config = rollout_learned_everyone_mask_vs_all_human_config
    elif config_case == 11:
        config = rollout_handcrafted_vs_all_human_config
    elif config_case == 12:
        config = rollout_default_vs_all_human_config
    elif config_case == 13:
        config = rollout_all_human_vs_learned_config
    elif config_case == 14:
        config = rollout_all_human_vs_handcrafted_config
    elif config_case == 15:
        config = rollout_all_human_vs_default_config
    #elif config_case == 13:
    #    config = rollout_no_history_learned_vs_all_human_config

    if config_case >= 6:
        cur_run_similarity_plots_path = \
            similarity_plots_path / rollout_learned_no_mask_load_data_option.custom_rollout_extension
    else:
        cur_run_similarity_plots_path = similarity_plots_path

    os.makedirs(cur_run_similarity_plots_path, exist_ok=True)
    similarity_df = load_hdf5_to_pd(config.similarity_data_path)
    # remove matches to self
    similarity_df = similarity_df[similarity_df[dtw_cost_col] != 0.]
    # need to load this early for filtering
    predicted_data = LoadDataResult(config.predicted_load_data_options)
    similarity_match_index_df = load_hdf5_to_pd(config.similarity_data_path, root_key='extra')

    start_similarity_plot_time = time.perf_counter()
    plot_trajectory_comparison_histograms(similarity_df, config, cur_run_similarity_plots_path)
    end_similarity_plot_time = time.perf_counter()
    print(f"similarity plot time {end_similarity_plot_time - start_similarity_plot_time: 0.4f}")

    if just_plot_summaries and not plot_trajectories:
        return

    # computing mapping between predict and ground truth
    # multiple predicted rounds may match to same ground truth round, don't save them multiple times
    start_predicted_to_ground_truth_time = time.perf_counter()
    predicted_to_ground_truth_dict = build_predicted_to_ground_truth_dict(similarity_df)
    end_predicted_to_ground_truth_time = time.perf_counter()
    print(f"predicted to ground truth time {end_predicted_to_ground_truth_time - start_predicted_to_ground_truth_time: 0.4f}")

    # load data
    start_predicted_load_time = time.perf_counter()
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

    trajectory_plots = None
    if plot_trajectories:
        start_heatmaps_plot_time = time.perf_counter()
        trajectory_plots = plot_trajectory_comparisons(similarity_df, predicted_model, ground_truth_model,
                                                       config, cur_run_similarity_plots_path,
                                                       debug_caching_override=config_case >= 13)
        end_heatmaps_plot_time = time.perf_counter()
        print(f"heatmaps plot time {end_heatmaps_plot_time - start_heatmaps_plot_time: 0.4f}")

    if just_plot_summaries:
        return trajectory_plots

    vis_two(predicted_model, ground_truth_model, predicted_to_ground_truth_dict, similarity_match_index_df)


if __name__ == "__main__":
    config_cases_str = sys.argv[2]
    trajectory_plots_by_player_type = []
    for config_case_str in config_cases_str.split(','):
        trajectory_plots_by_player_type.append(compare_trajectories(int(config_case_str)))
    if len(trajectory_plots_by_player_type) > 1 and trajectory_plots_by_player_type[0] is not None:
        concat_trajectory_plots_across_player_type(
            trajectory_plots_by_player_type,
            similarity_plots_path / rollout_learned_no_mask_load_data_option.custom_rollout_extension)
