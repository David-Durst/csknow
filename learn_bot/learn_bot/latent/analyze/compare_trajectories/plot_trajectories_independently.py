import dataclasses
import os
import sys
from typing import List, Tuple

import pandas as pd
from PIL import Image

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_and_events import plot_trajectory_dfs_and_event
from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_from_comparison import \
    concat_horizontal
from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import set_pd_print_options, \
    ComparisonConfig
from learn_bot.latent.analyze.compare_trajectories.region_constraints.compute_constraint_metrics import check_constraint_metrics, \
    ConstraintResult
from learn_bot.latent.analyze.compare_trajectories.run_trajectory_comparison import rollout_load_data_option, rollout_handcrafted_vs_all_human_config
from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path
from learn_bot.latent.engagement.column_names import round_id_column, round_number_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.column_names import round_test_name_col
from learn_bot.latent.place_area.load_data import LoadDataResult, LoadDataOptions

rollout_no_mask_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_11_5_23_prebaked_no_mask_100_tries")
rollout_no_mask_config = dataclasses.replace(rollout_handcrafted_vs_all_human_config,
                                             predicted_load_data_options=rollout_no_mask_load_data_option,
                                             metric_cost_title="Rollout Learned No Mask Randomized vs")

rollout_everyone_mask_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_10_27_23_prebaked_everyone_mask_randomized")
rollout_everyone_mask_config = dataclasses.replace(rollout_handcrafted_vs_all_human_config,
                                                   predicted_load_data_options=rollout_everyone_mask_load_data_option,
                                                   metric_cost_title="Rollout Learned Everyone Mask Randomized vs")

rollout_no_mask_not_randomized_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_10_26_23_prebaked_no_mask_not_randomized")
rollout_no_mask_not_randomized_config = dataclasses.replace(rollout_handcrafted_vs_all_human_config,
                                             predicted_load_data_options=rollout_no_mask_load_data_option,
                                             metric_cost_title="Rollout Learned No Mask Not Randomized vs")


def plot_trajectories_for_one_config(load_data_options: LoadDataOptions, config: ComparisonConfig) -> \
        Tuple[List[Image.Image], List[ConstraintResult],  List[str]]:
    print(f"handling {load_data_options.custom_rollout_extension}")
    set_pd_print_options()
    img_result: List[Image] = []
    constraint_result: List[ConstraintResult] = []

    # load data
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result)
    per_round_df = loaded_model.cur_loaded_df[[round_id_column, round_number_column]] \
        .groupby([round_id_column], as_index=False).min()
    round_test_names_with_invalid = loaded_model.get_round_test_names().tolist()
    round_test_names = [round_test_names_with_invalid[i] for i in per_round_df[round_id_column].to_list()]
    round_test_names_no_numbers = [''.join([i for i in n if not i.isdigit()]) for n in round_test_names]
    assert len([s for s in round_test_names_no_numbers if 'INVALID' in s]) == 0
    per_round_df[round_test_name_col] = round_test_names_no_numbers
    unique_round_test_names = per_round_df[round_test_name_col].unique()

    for round_test_name in unique_round_test_names:
        print("plotting " + round_test_name)
        round_ids = per_round_df[per_round_df[round_test_name_col] == round_test_name][round_id_column]
        round_trajectory_dfs: List[pd.DataFrame] = []
        for round_id in round_ids:
            round_trajectory_dfs.append(loaded_model.cur_loaded_df[loaded_model.cur_loaded_df[round_id_column] ==
                                                                   round_id])
        img_result.append(plot_trajectory_dfs_and_event(round_trajectory_dfs, config, True, True, True, plot_starts=True))
        constraint_result.append(check_constraint_metrics(round_trajectory_dfs, round_test_name, None, img_result[-1]))

    return img_result, constraint_result, list(unique_round_test_names)


def run_independent_trajectory_vis():
    config_cases_str = sys.argv[2]
    trajectory_plots_by_config = []
    trajectory_plots_names = []

    plots_path = similarity_plots_path / rollout_no_mask_load_data_option.custom_rollout_extension
    os.makedirs(plots_path, exist_ok=True)

    for config_case_str in config_cases_str.split(','):
        config_case = int(config_case_str)

        if config_case == 0:
            data_option = rollout_no_mask_load_data_option
            config = rollout_no_mask_config
        elif config_case == 1:
            data_option = rollout_everyone_mask_load_data_option
            config = rollout_everyone_mask_config
        elif config_case == 2:
            data_option = rollout_no_mask_not_randomized_load_data_option
            config = rollout_no_mask_not_randomized_config
        else:
            print(f"invalid config case: {config_case}")
            exit(0)
        if len(sys.argv) == 4:
            data_option = dataclasses.replace(data_option,
                                              custom_rollout_extension=sys.argv[3])
            config = dataclasses.replace(config, predicted_load_data_options=data_option)
        result = plot_trajectories_for_one_config(data_option, config)
        trajectory_plots_by_config.append(result[0])
        for constraint_result in result[1]:
            constraint_result.save(plots_path / f'{constraint_result.test_name}.csv', config.metric_cost_title)
        trajectory_plots_names = result[2]

    trajectory_plots_by_test_name = [[p[i] for p in trajectory_plots_by_config] for i in range(len(trajectory_plots_names))]
    image_per_test_name = [concat_horizontal(plots) for plots in trajectory_plots_by_test_name]
    for img, test_name in zip(image_per_test_name, trajectory_plots_names):
        img.save(plots_path / f'{test_name}.png')


if __name__ == "__main__":
    run_independent_trajectory_vis()
