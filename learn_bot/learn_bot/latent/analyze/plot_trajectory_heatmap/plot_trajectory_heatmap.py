import dataclasses
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import get_hdf5_to_round_ids
from learn_bot.latent.analyze.compare_trajectories.run_trajectory_comparison import rollout_load_data_option
from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.plot_trajectory_heatmap.plot_one_trajectory_np import plot_one_trajectory_np, \
    plot_trajectories_to_image
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.load_data import LoadDataResult
import learn_bot.latent.vis.run_vis_checkpoint as run_vis_checkpoint


human_title_str = 'Human'


def run_one_dataset_trajectory_heatmap(use_all_human_data: bool, rollout_extension: str):
    if use_all_human_data:
        load_data_options = run_vis_checkpoint.load_data_options
        title_str = human_title_str
    else:
        load_data_options = dataclasses.replace(rollout_load_data_option,
                                                custom_rollout_extension=rollout_extension + '*')
        title_str = rollout_extension

    # load data
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result)
    hdf5_to_round_ids = get_hdf5_to_round_ids(load_data_result)[0]

    with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
        for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
            if use_all_human_data:
                trajectory_filter_options = \
                    TrajectoryFilterOptions(set(hdf5_to_round_ids[str(hdf5_wrapper.hdf5_path.name)]))
            else:
                trajectory_filter_options = TrajectoryFilterOptions(None)

            loaded_model.cur_hdf5_index = i
            loaded_model.load_cur_dataset_only(include_outputs=False)

            plot_one_trajectory_np(loaded_model, loaded_model.get_cur_id_df(), loaded_model.cur_dataset.X,
                                   trajectory_filter_options, title_str)
            pbar.update(1)


def run_trajectory_heatmaps():
    rollout_extensions = sys.argv[2].split(',')
    plots_path = similarity_plots_path / rollout_extensions[0]
    os.makedirs(plots_path, exist_ok=True)

    run_one_dataset_trajectory_heatmap(True, '')
    if rollout_extensions[0] != 'invalid':
        for rollout_extension in rollout_extensions:
            run_one_dataset_trajectory_heatmap(False, rollout_extension)

    title_strs = [human_title_str] + (rollout_extensions if rollout_extensions[0] != 'invalid' else [])
    plot_trajectories_to_image(title_strs, True, plots_path)


if __name__ == "__main__":
    run_trajectory_heatmaps()
