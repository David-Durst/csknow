import dataclasses
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import get_hdf5_to_round_ids
from learn_bot.latent.analyze.compare_trajectories.run_trajectory_comparison import rollout_load_data_option
from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions, \
    region_constraints
from learn_bot.latent.analyze.plot_trajectory_heatmap.plot_one_trajectory_np import plot_one_trajectory_dataset, \
    plot_trajectories_to_image, clear_title_caches
from learn_bot.latent.load_model import load_model_file, LoadedModel
from learn_bot.latent.place_area.load_data import LoadDataResult
import learn_bot.latent.vis.run_vis_checkpoint as run_vis_checkpoint


human_title_str = 'Human'
title_to_loaded_model: Dict[str, LoadedModel] = {}
title_to_hdf5_to_round_ids: Dict[str, Dict[str, List[int]]] = {}


def run_one_dataset_trajectory_heatmap(use_all_human_data: bool, title: str,
                                       base_trajectory_filter_options: TrajectoryFilterOptions,
                                       push_only_human_data: bool = True):
    print(f"{title} {str(base_trajectory_filter_options)}")
    
    if use_all_human_data:
        load_data_options = run_vis_checkpoint.load_data_options
    else:
        load_data_options = dataclasses.replace(rollout_load_data_option,
                                                custom_rollout_extension='_' + title + '*')

    # load data
    if title in title_to_loaded_model:
        loaded_model = title_to_loaded_model[title]
        hdf5_to_round_ids = title_to_hdf5_to_round_ids[title]
    else:
        load_data_result = LoadDataResult(load_data_options)
        loaded_model = load_model_file(load_data_result)
        title_to_loaded_model[title] = loaded_model
        hdf5_to_round_ids = get_hdf5_to_round_ids(load_data_result, push_only=push_only_human_data)[0]
        title_to_hdf5_to_round_ids[title] = hdf5_to_round_ids

    with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
        for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
            if use_all_human_data:
                trajectory_filter_options = \
                    dataclasses.replace(base_trajectory_filter_options,
                                        valid_round_ids=set(hdf5_to_round_ids[str(hdf5_wrapper.hdf5_path.name)]))
            else:
                trajectory_filter_options = base_trajectory_filter_options

            loaded_model.cur_hdf5_index = i
            loaded_model.load_cur_dataset_only(include_outputs=False)

            plot_one_trajectory_dataset(loaded_model, loaded_model.get_cur_id_df(), loaded_model.cur_dataset.X,
                                        trajectory_filter_options, title)
            pbar.update(1)


def run_trajectory_heatmaps_one_filter_option(trajectory_filter_options: TrajectoryFilterOptions,
                                              rollout_extensions: List[str], plots_path: Path):
    clear_title_caches()

    run_one_dataset_trajectory_heatmap(True, human_title_str, trajectory_filter_options)
    if rollout_extensions[0] == 'invalid':
        run_one_dataset_trajectory_heatmap(True, human_title_str + ' Save', trajectory_filter_options,
                                           push_only_human_data=False)
    else:
        for rollout_extension in rollout_extensions:
            run_one_dataset_trajectory_heatmap(False, rollout_extension, trajectory_filter_options)

    title_strs = [human_title_str] + (rollout_extensions if rollout_extensions[0] != 'invalid'
                                      else [human_title_str + ' Save'])
    plot_trajectories_to_image(title_strs, True, plots_path, trajectory_filter_options)


def run_trajectory_heatmaps():
    rollout_extensions = sys.argv[2].split(',')
    plots_path = similarity_plots_path / rollout_extensions[0]
    os.makedirs(plots_path, exist_ok=True)

    run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(),
                                              rollout_extensions, plots_path)

    for region_constraint_str, region_constraint in region_constraints.items():
        run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(player_starts_in_region=region_constraint,
                                                                          region_name=region_constraint_str),
                                                  rollout_extensions, plots_path)

    for round_game_seconds in [range(0, 5), range(5, 10), range(10, 15), range(15, 20), range(20, 25), range(25, 30)]:
        run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(round_game_seconds=round_game_seconds),
                                                  rollout_extensions, plots_path)


if __name__ == "__main__":
    run_trajectory_heatmaps()
