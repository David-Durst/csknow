import os
import pickle
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import ComparisonConfig
from learn_bot.latent.analyze.run_coverage import coverage_pickle_path
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.transformer_nested_hidden_latent_model import d2_min, d2_max

max_distance = 1e7

def extra_data_from_metric_title(metric_title: str, predicted: bool) -> str:
    if predicted:
        start_str = "Rollout "
        end_str = " vs"
    else:
        start_str = "vs "
        end_str = " Distribution"

    start_index = metric_title.index(start_str) + len(start_str)
    end_index = metric_title.index(end_str)

    return metric_title[start_index:end_index] + (" All Data" if predicted else " Most Similar")


def plot_occupancy_heatmap(trajectory_dfs: List[pd.DataFrame], config: ComparisonConfig, distance_to_other_player: bool,
                           teammate: bool, similarity_plots_path: Optional[Path],
                           valid_players_dfs: List[pd.DataFrame] = [],
                           title_text: Optional[str] = None) -> Optional[Image.Image]:
    counts_heatmap = None
    sums_heatmap = None
    #x_bins = None
    #y_bins = None

    with open(coverage_pickle_path, "rb") as infile:
        (coverage_heatmap, x_bins, y_bins) = pickle.load(infile)
    #empty_removed = np.where(coverage_heatmap < 0.5, -1., coverage_heatmap)
    #zeros_in_valid = np.where(empty_removed > 0.5, 0., empty_removed)
    #counts_heatmap = zeros_in_valid
    #sums_heatmap = zeros_in_valid.copy()

    if len(valid_players_dfs) == 0:
        valid_players_dfs = [pd.DataFrame() for _ in trajectory_dfs]

    trajectory_df = pd.concat(trajectory_dfs)
    valid_players_df = pd.concat(valid_players_dfs)

    for player_place_area_columns in specific_player_place_area_columns:
        # make sure player is alive if don't have another condition
        if len(valid_players_df) == 0:
            cur_player_trajectory_df = trajectory_df[trajectory_df[player_place_area_columns.alive] == 1]
            if cur_player_trajectory_df.empty:
                continue
        else:
            cur_player_trajectory_df = trajectory_df[valid_players_df[player_place_area_columns.player_id]]

        # iterate over other players to find closest
        conditional_distances: Dict[str, pd.Series] = {}
        for other_player_place_area_columns in specific_player_place_area_columns:
            # don't match to same player
            if other_player_place_area_columns.player_id == player_place_area_columns.player_id:
                continue
            # condition for counting is alive and same team (if plotting nearest teammate) or other team
            # (if plotting nearest enemy)
            other_condition = cur_player_trajectory_df[other_player_place_area_columns.alive] == 1
            if teammate:
                other_condition = other_condition & \
                                  (cur_player_trajectory_df[other_player_place_area_columns.ct_team] ==
                                   cur_player_trajectory_df[player_place_area_columns.ct_team])
            else:
                other_condition = other_condition & \
                                  (cur_player_trajectory_df[other_player_place_area_columns.ct_team] !=
                                   cur_player_trajectory_df[player_place_area_columns.ct_team])

            delta_x = (cur_player_trajectory_df[player_place_area_columns.pos[0]] -
                       cur_player_trajectory_df[other_player_place_area_columns.pos[0]]) ** 2.
            delta_y = (cur_player_trajectory_df[player_place_area_columns.pos[1]] -
                       cur_player_trajectory_df[other_player_place_area_columns.pos[1]]) ** 2.
            delta_z = (cur_player_trajectory_df[player_place_area_columns.pos[2]] -
                       cur_player_trajectory_df[other_player_place_area_columns.pos[2]]) ** 2.
            distance = (delta_x + delta_y + delta_z).pow(0.5)
            conditional_distances[other_player_place_area_columns.player_id] = \
                distance.where(other_condition, max_distance)

        # compute values for this trajectory in heatmap
        conditional_distances_df = pd.DataFrame(conditional_distances)

        #if len(valid_players_df) != 0:
        #    if len(conditional_distances_df.index) != len(valid_players_df.index):
        #        print('dude')
        #    if not (conditional_distances_df.index == valid_players_df.index).all():
        #        print('hi')
        #    conditional_distances_df = \
        #        conditional_distances_df[valid_players_df[player_place_area_columns.player_id]]

        player_xy_pos_distance_df = pd.DataFrame({
            "x pos": cur_player_trajectory_df[player_place_area_columns.pos[0]],
            "y pos": cur_player_trajectory_df[player_place_area_columns.pos[1]],
            "distance": conditional_distances_df.min(axis=1)
        })
        player_xy_pos_distance_df = player_xy_pos_distance_df[player_xy_pos_distance_df['distance'] <
                                                              max_distance / 10.]
        if player_xy_pos_distance_df.empty:
            continue
        player_min_distances_to_other = player_xy_pos_distance_df["distance"].to_numpy()
        x_pos = player_xy_pos_distance_df["x pos"].to_numpy()
        y_pos = player_xy_pos_distance_df["y pos"].to_numpy()

        # add to heatmap bins
        if counts_heatmap is None:
            counts_heatmap, _, _ = np.histogram2d(x_pos, y_pos, bins=[x_bins, y_bins])
            sums_heatmap, _, _ = np.histogram2d(x_pos, y_pos, weights=player_min_distances_to_other,
                                                bins=[x_bins, y_bins])
        else:
            tmp_counts_heatmap, _, _ = np.histogram2d(x_pos, y_pos, bins=[x_bins, y_bins])
            counts_heatmap += tmp_counts_heatmap
            tmp_sums_heatmap, _, _ = np.histogram2d(x_pos, y_pos, weights=player_min_distances_to_other,
                                                    bins=[x_bins, y_bins])
            sums_heatmap += tmp_sums_heatmap

    counts_heatmap = np.ma.masked_where(coverage_heatmap < 0.5, counts_heatmap)
    sums_heatmap = np.ma.masked_where(coverage_heatmap < 0.5, sums_heatmap)

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    if title_text is None:
        teammate_text = "Teammate" if teammate else "Enemy"
        fig.suptitle(extra_data_from_metric_title(config.metric_cost_title, True) + " Distance To " + teammate_text,
                     fontsize=16)
    else:
        fig.suptitle(title_text, fontsize=16)
    ax = fig.subplots(1, 1)

    counts_heatmap = counts_heatmap.T
    sums_heatmap = sums_heatmap.T

    grid_x, grid_y = np.meshgrid(x_bins, y_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_heatmap = sums_heatmap / counts_heatmap
    non_nan_min = np.nanmin(avg_heatmap)
    non_nan_max = np.nanmax(avg_heatmap)
    avg_heatmap[np.isnan(avg_heatmap)] = 0

    if non_nan_min > 100.:
        non_nan_min = 100.
    elif non_nan_min > 10.:
        non_nan_min = 10.
    else:
        non_nan_min = 1.

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_under('gray')

    if distance_to_other_player:
        heatmap_im = ax.pcolormesh(grid_x, grid_y, avg_heatmap,
                                   norm=LogNorm(vmin=non_nan_min, vmax=non_nan_max),
                                   #vmin=non_nan_min,
                                   #vmax=non_nan_max,
                                   #norm=TwoSlopeNorm(vmin=non_nan_min, vcenter=3000, vmax=non_nan_max),
                                   cmap='viridis')
    else:
        heatmap_im = ax.pcolormesh(grid_x, grid_y, counts_heatmap,
                                   #norm=LogNorm(vmin=non_nan_min, vmax=non_nan_max),
                                   vmin=1,
                                   #vmax=np.max(counts_heatmap),
                                   # norm=TwoSlopeNorm(vmin=non_nan_min, vcenter=3000, vmax=non_nan_max),
                                   cmap=cmap)
    cbar = fig.colorbar(heatmap_im, ax=ax)
    if distance_to_other_player:
        cbar.ax.set_ylabel('Mean Distance To ' + teammate_text, rotation=270, labelpad=15, fontsize=14)
    else:
        cbar.ax.set_ylabel('Number of Per-Player Data Points', rotation=270, labelpad=15, fontsize=14)

    ## Get the default ticks and tick labels
    #ticklabels = cbar.ax.get_ymajorticklabels()
    #ticks = list(cbar.get_ticks())

    ## Append the ticks (and their labels) for minimum and the maximum value
    #cbar.set_ticks([non_nan_min, non_nan_max] + ticks)
    #cbar.set_ticklabels([non_nan_min, non_nan_max] + ticklabels)

    if similarity_plots_path is None:
        tmp_dir = tempfile.gettempdir()
        tmp_file = Path(tmp_dir) / 'tmp_heatmap.png'
        plt.savefig(tmp_file)
        img = Image.open(tmp_file)
        img.load()
        tmp_file.unlink()
        return img
    else:
        plt.savefig(similarity_plots_path / (config.metric_cost_file_name + '_distance_' + teammate_text.lower() + '.png'))
        return None
