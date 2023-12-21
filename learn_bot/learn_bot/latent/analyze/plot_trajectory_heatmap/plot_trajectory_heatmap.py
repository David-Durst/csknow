import dataclasses
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import get_hdf5_to_test_round_ids, \
    get_hdf5_to_round_ids_fresh
from learn_bot.latent.analyze.compare_trajectories.run_trajectory_comparison import rollout_load_data_option
from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path
from learn_bot.latent.analyze.plot_trajectory_heatmap.compute_earth_mover_distance import \
    compute_trajectory_earth_mover_distances
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions, \
    region_constraints
from learn_bot.latent.analyze.plot_trajectory_heatmap.render_diffs import plot_trajectory_diffs_to_image
from learn_bot.latent.analyze.plot_trajectory_heatmap.render_heatmaps import plot_trajectories_to_image
from learn_bot.latent.analyze.plot_trajectory_heatmap.render_key_event_heatmap import plot_key_event_heatmaps
from learn_bot.latent.analyze.plot_trajectory_heatmap.compute_metrics import compute_metrics
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import plot_one_trajectory_dataset, \
    clear_title_caches, get_title_to_team_to_key_event_pos
from learn_bot.latent.analyze.plot_trajectory_heatmap.plot_points_per_game_seconds_bar_chart import \
    record_points_per_one_game_seconds_range, plot_points_per_game_seconds, reset_points_per_game_seconds_state
from learn_bot.latent.load_model import load_model_file, LoadedModel
from learn_bot.latent.place_area.load_data import LoadDataResult
import learn_bot.latent.vis.run_vis_checkpoint as run_vis_checkpoint


human_title_str = 'Human'
title_to_loaded_model: Dict[str, LoadedModel] = {}
title_to_hdf5_to_round_ids: Dict[str, Dict[str, List[int]]] = {}


# if push_only is true, get only pushes. if false, get saves only
def run_one_dataset_trajectory_heatmap(use_all_human_data: bool, title: str,
                                       base_trajectory_filter_options: TrajectoryFilterOptions,
                                       push_only_human_data: bool = True,
                                       # limit to hand labeled push/save data for first hdf5 file
                                       plot_only_first_hdf5_file_train_and_test: bool = False):
    print(f"{title} {str(base_trajectory_filter_options)}")
    
    if use_all_human_data:
        load_data_options = run_vis_checkpoint.load_data_options
    else:
        load_data_options = dataclasses.replace(rollout_load_data_option,
                                                custom_rollout_extension='_' + title + '*')

    # load data, caching it so don't repeatedly load data
    if title in title_to_loaded_model:
        loaded_model = title_to_loaded_model[title]
        hdf5_to_round_ids = title_to_hdf5_to_round_ids[title]
    else:
        load_data_result = LoadDataResult(load_data_options)
        loaded_model = load_model_file(load_data_result)
        title_to_loaded_model[title] = loaded_model
        if plot_only_first_hdf5_file_train_and_test:
            hdf5_to_round_ids = get_hdf5_to_round_ids_fresh(load_data_result, False,
                                                            push_only=push_only_human_data,
                                                            save_only=not push_only_human_data)[0]
        else:
            hdf5_to_round_ids = get_hdf5_to_test_round_ids(push_only=push_only_human_data)[0]
        title_to_hdf5_to_round_ids[title] = hdf5_to_round_ids

    with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
        for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
            if plot_only_first_hdf5_file_train_and_test and i > 0:
                break
            if use_all_human_data:
                hdf5_key = str(hdf5_wrapper.hdf5_path.name)
                if hdf5_key not in hdf5_to_round_ids:
                    #print(f'skipping {hdf5_key}')
                    continue
                trajectory_filter_options = \
                    dataclasses.replace(base_trajectory_filter_options, valid_round_ids=set(hdf5_to_round_ids[hdf5_key]))
            else:
                trajectory_filter_options = base_trajectory_filter_options

            loaded_model.cur_hdf5_index = i
            loaded_model.load_cur_dataset_only(include_outputs=False)

            plot_one_trajectory_dataset(loaded_model, loaded_model.get_cur_id_df(), loaded_model.get_cur_vis_df(),
                                        loaded_model.cur_dataset.X, trajectory_filter_options, title)
            pbar.update(1)


def run_trajectory_heatmaps_one_filter_option(trajectory_filter_options: TrajectoryFilterOptions,
                                              rollout_extensions: List[str], diff_indices: List[int], plots_path: Path,
                                              plot_only_first_hdf5_file_train_and_test: bool):
    clear_title_caches()

    run_one_dataset_trajectory_heatmap(True, human_title_str, trajectory_filter_options,
                                       plot_only_first_hdf5_file_train_and_test=plot_only_first_hdf5_file_train_and_test)
    if rollout_extensions[0] == 'invalid':
        run_one_dataset_trajectory_heatmap(True, human_title_str + ' Save', trajectory_filter_options,
                                           push_only_human_data=False,
                                           plot_only_first_hdf5_file_train_and_test=plot_only_first_hdf5_file_train_and_test)
    else:
        for rollout_extension in rollout_extensions:
            run_one_dataset_trajectory_heatmap(False, rollout_extension, trajectory_filter_options)

    title_strs = [human_title_str] + (rollout_extensions if rollout_extensions[0] != 'invalid'
                                      else [human_title_str + ' Save'])
    # run this before plotting trajectories so get unscaled buffers
    compute_trajectory_earth_mover_distances(title_strs, diff_indices, plots_path, trajectory_filter_options)
    plot_trajectories_to_image(title_strs, True, plots_path, trajectory_filter_options)
    plot_trajectory_diffs_to_image(title_strs, diff_indices, plots_path, trajectory_filter_options)
    compute_metrics(trajectory_filter_options, plots_path)
    plot_key_event_heatmaps(get_title_to_team_to_key_event_pos(), trajectory_filter_options, plots_path)


def run_trajectory_heatmaps(plot_only_first_hdf5_file_train_and_test: bool):
    rollout_extensions = sys.argv[2].split(',')
    # indices for plots to diff with (first plot is human data, which all rollout extensions come after, so offset indexing by 1)
    if len(sys.argv) == 4:
        diff_indices = [int(x) for x in sys.argv[3].split(',')]
    elif rollout_extensions[0] == 'invalid':
        diff_indices = [0 for _ in range(len(rollout_extensions))]
    else:
        raise Exception('must provide diff indices if not just plotting human data')
    plots_path = similarity_plots_path / (rollout_extensions[0] +
                                          ("_all_first" if plot_only_first_hdf5_file_train_and_test else ""))
    os.makedirs(plots_path, exist_ok=True)

    os.makedirs(plots_path / 'diff', exist_ok=True)

    run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(compute_speeds=True, compute_lifetimes=True,
                                                                      compute_shots_per_kill=True),
                                              rollout_extensions, diff_indices, plots_path,
                                              plot_only_first_hdf5_file_train_and_test)

    # plot distributions during key events
    run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(compute_speeds=True, only_kill=True),
                                              rollout_extensions, diff_indices, plots_path,
                                              plot_only_first_hdf5_file_train_and_test)

    run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(compute_speeds=True, only_killed=True),
                                              rollout_extensions, diff_indices, plots_path,
                                              plot_only_first_hdf5_file_train_and_test)

    run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(compute_speeds=True, only_killed_or_end=True),
                                              rollout_extensions, diff_indices, plots_path,
                                              plot_only_first_hdf5_file_train_and_test)

    run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(compute_speeds=True, only_shots=True),
                                              rollout_extensions, diff_indices, plots_path,
                                              plot_only_first_hdf5_file_train_and_test)

    for region_constraint_str, region_constraint in region_constraints.items():
        # haven't had luck seeing anything interesting with just one player
        #run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(player_starts_in_region=region_constraint,
        #                                                                  region_name=region_constraint_str,
        #                                                                  include_all_players_when_one_in_region=False),
        #                                          rollout_extensions, diff_indices, plots_path,
        #                                          plot_only_first_hdf5_file_train_and_test)
        run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(player_starts_in_region=region_constraint,
                                                                          region_name=region_constraint_str,
                                                                          include_all_players_when_one_in_region=True),
                                                  rollout_extensions, diff_indices, plots_path,
                                                  plot_only_first_hdf5_file_train_and_test)

    reset_points_per_game_seconds_state()
    for round_game_seconds in [range(0, 5), range(5, 10), range(10, 15), range(15, 20), range(20, 25), range(25, 30)]:
        run_trajectory_heatmaps_one_filter_option(TrajectoryFilterOptions(round_game_seconds=round_game_seconds),
                                                  rollout_extensions, diff_indices, plots_path,
                                                  plot_only_first_hdf5_file_train_and_test)
        record_points_per_one_game_seconds_range(round_game_seconds)
    plot_points_per_game_seconds(plots_path)


if __name__ == "__main__":
    run_trajectory_heatmaps(False)
    if sys.argv[2] == 'invalid':
        title_to_loaded_model = {}
        title_to_hdf5_to_round_ids = {}
        run_trajectory_heatmaps(True)
