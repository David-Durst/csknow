from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import ComparisonConfig
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



def plot_distance_to_teammate_enemy_from_trajectory_dfs(trajectory_dfs: List[pd.DataFrame], config: ComparisonConfig,
                                                        teammate: bool, similarity_plots_path: Path):
    counts_heatmap = None
    sums_heatmap = None
    x_bins = None
    y_bins = None

    for trajectory_df in trajectory_dfs:
        # since this was split with : rather than _, need to remove last _
        for player_place_area_columns in specific_player_place_area_columns:
            # make sure player is alive
            cur_player_trajectory_df = trajectory_df[trajectory_df[player_place_area_columns.alive] == 1]
            if cur_player_trajectory_df.empty:
                continue

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
            if x_bins is None:
                counts_heatmap, x_bins, y_bins = np.histogram2d(x_pos, y_pos, bins=125,
                                                                         range=[[d2_min[0], d2_max[0]],
                                                                                [d2_min[1], d2_max[1]]])
                sums_heatmap, _, _ = np.histogram2d(x_pos, y_pos, weights=player_min_distances_to_other,
                                                    bins=[x_bins, y_bins])
            else:
                tmp_counts_heatmap, _, _ = np.histogram2d(x_pos, y_pos, bins=[x_bins, y_bins])
                counts_heatmap += tmp_counts_heatmap
                tmp_sums_heatmap, _, _ = np.histogram2d(x_pos, y_pos, weights=player_min_distances_to_other,
                                                        bins=[x_bins, y_bins])
                sums_heatmap += tmp_sums_heatmap

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    teammate_text = "Teammate" if teammate else "Enemy"
    fig.suptitle(extra_data_from_metric_title(config.metric_cost_title, True) + " Distance To " + teammate_text,
                 fontsize=16)
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

    heatmap_im = ax.pcolormesh(grid_x, grid_y, avg_heatmap,
                               norm=LogNorm(vmin=non_nan_min, vmax=non_nan_max),
                               #vmin=non_nan_min,
                               #vmax=non_nan_max,
                               #norm=TwoSlopeNorm(vmin=non_nan_min, vcenter=3000, vmax=non_nan_max),
                               cmap='viridis')
    cbar = fig.colorbar(heatmap_im, ax=ax)
    cbar.ax.set_ylabel('Mean Distance To ' + teammate_text, rotation=270, labelpad=15, fontsize=14)

    ## Get the default ticks and tick labels
    #ticklabels = cbar.ax.get_ymajorticklabels()
    #ticks = list(cbar.get_ticks())

    ## Append the ticks (and their labels) for minimum and the maximum value
    #cbar.set_ticks([non_nan_min, non_nan_max] + ticks)
    #cbar.set_ticklabels([non_nan_min, non_nan_max] + ticklabels)

    plt.savefig(similarity_plots_path / (config.metric_cost_file_name + '_distance_' + teammate_text.lower() + '.png'))
