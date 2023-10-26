import dataclasses
import sys
from pathlib import Path
from typing import Set, List

import pandas as pd
from PIL import Image

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_and_events import plot_trajectory_dfs_and_event
from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_from_comparison import \
    select_trajectories_into_dfs, RoundsForTrajectorySelection, TrajectoryPlots, concat_horizontal, concat_vertical
from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import set_pd_print_options, \
    ComparisonConfig
from learn_bot.latent.analyze.compare_trajectories.run_trajectory_comparison import all_human_load_data_option, \
    rollout_load_data_option, rollout_handcrafted_vs_all_human_config
from learn_bot.latent.analyze.comparison_column_names import rollout_handcrafted_vs_all_human_similarity_hdf5_data_path, \
    similarity_plots_path
from learn_bot.latent.engagement.column_names import round_id_column, round_number_column
from learn_bot.latent.load_model import LoadedModel, load_model_file
from learn_bot.latent.place_area.column_names import round_test_name_col
from learn_bot.latent.place_area.load_data import LoadDataResult, LoadDataOptions

rollout_no_mask_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_10_25_23_prebaked_no_mask")
rollout_no_mask_config = dataclasses.replace(rollout_handcrafted_vs_all_human_config,
                                             predicted_load_data_options=rollout_no_mask_load_data_option,
                                             metric_cost_title="Learned No Mask")

rollout_everyone_mask_load_data_option = dataclasses.replace(rollout_load_data_option,
                                                       custom_rollout_extension= "_10_25_23_prebaked_everyone_mask_")
rollout_everyone_mask_config = dataclasses.replace(rollout_handcrafted_vs_all_human_config,
                                                   predicted_load_data_options=rollout_everyone_mask_load_data_option,
                                                   metric_cost_title="Learned Everyone Mask")



def plot_trajectories_for_one_config(load_data_options: LoadDataOptions, config: ComparisonConfig) -> List[Image]:
    print(f"handling {load_data_options.custom_rollout_extension}")
    set_pd_print_options()
    result: List[Image] = []

    # load data
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result)
    round_test_names = loaded_model.get_round_test_names().tolist()
    round_test_names_no_numbers = [''.join([i for i in n if not i.isdigit()]) for n in round_test_names]
    per_round_df = loaded_model.cur_loaded_df.groupby([round_id_column]).min(round_number_column)
    assert len(per_round_df) == len(round_test_names_no_numbers)
    per_round_df[round_test_name_col] = round_test_names_no_numbers
    unique_round_test_names = per_round_df[round_test_name_col].unique()

    for round_test_name in unique_round_test_names:
        print("plotting " + round_test_name)
        round_ids = per_round_df[per_round_df[round_test_name_col] == round_test_name][round_id_column]
        round_trajectory_dfs: List[pd.DataFrame] = []
        for round_id in round_ids:
            round_trajectory_dfs.append(loaded_model.cur_loaded_df[loaded_model.cur_loaded_df[round_id_column] ==
                                                                   round_id])
        result.append(plot_trajectory_dfs_and_event(round_trajectory_dfs, config, True, True, True,
                                                    title_appendix=config.metric_cost_title))

    return result


def run_independent_trajectory_vis():
    config_cases_str = sys.argv[2]
    trajectory_plots_by_config = []
    for config_case_str in config_cases_str.split(','):
        config_case = int(config_case_str)

        if config_case == 0:
            data_option = rollout_no_mask_load_data_option
            config = rollout_no_mask_config
        elif config_case == 1:
            data_option = rollout_everyone_mask_load_data_option
            config = rollout_everyone_mask_config
        else:
            print(f"invalid config case: {config_case}")
            exit(0)
        trajectory_plots_by_config.append(plot_trajectories_for_one_config(data_option, config))

    plots_path = similarity_plots_path / rollout_no_mask_load_data_option.custom_rollout_extension
    image_per_config = [concat_vertical(plots) for plots in trajectory_plots_by_config]
    final_image = concat_horizontal(image_per_config)
    final_image.save(plots_path / 'final_independent.png')


if __name__ == "__main__":
    run_independent_trajectory_vis()
