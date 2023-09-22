import os
import pickle
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List, Set, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.filter_trajectory_key_events import FilterEventType, \
    filter_trajectory_by_key_events, KeyAreas
from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import set_pd_print_options, \
    ComparisonConfig
from learn_bot.latent.analyze.comparison_column_names import metric_type_col, dtw_cost_col, \
    best_fit_ground_truth_trace_batch_col, predicted_trace_batch_col, best_match_id_col, predicted_round_id_col, \
    best_fit_ground_truth_round_id_col, predicted_round_number_col, best_fit_ground_truth_round_number_col
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, convert_to_canvas_coordinates, \
    bot_ct_color_list, bot_t_color_list
from learn_bot.latent.engagement.column_names import round_id_column, round_number_column
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import AABB
from learn_bot.latent.transformer_nested_hidden_latent_model import d2_min, d2_max
from learn_bot.libs.vec import Vec3

plot_n_most_similar = 1


@dataclass(frozen=True, eq=True)
class RoundForComparisonHeatmap:
    hdf5_file_name: str
    round_id: int
    round_number: int


def select_trajectories_into_dfs(loaded_model: LoadedModel,
                                 rounds_for_comparison: List[RoundForComparisonHeatmap]) -> List[pd.DataFrame]:
    selected_dfs: List[pd.DataFrame] = []
    hdf5s_for_selection: Set[str] = {r.hdf5_file_name for r in rounds_for_comparison}

    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        print(f"Processing hdf5 {i + 1} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
        loaded_model.cur_hdf5_index = i
        wrapper_hdf5_file_name = hdf5_wrapper.hdf5_path.name

        if wrapper_hdf5_file_name not in hdf5s_for_selection:
            continue

        loaded_model.load_cur_hdf5_as_pd(load_cur_dataset=False, cast_bool_to_int=False)
        for round_for_comparison in rounds_for_comparison:
            if wrapper_hdf5_file_name == round_for_comparison.hdf5_file_name:
                selected_dfs.append(loaded_model.cur_loaded_df[loaded_model.cur_loaded_df[round_id_column] ==
                                                               round_for_comparison.round_id])
                assert selected_dfs[-1][round_number_column].iloc[0] == round_for_comparison.round_number

    return selected_dfs


title_font = ImageFont.truetype("arial.ttf", 25)


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


num_trajectories_for_color = 700


def plot_trajectory_dfs(trajectory_dfs: List[pd.DataFrame], config: ComparisonConfig, predicted: bool,
                        include_ct: bool, include_t: bool,
                        filter_event_type: Optional[FilterEventType] = None,
                        key_areas: Optional[KeyAreas] = None) -> Image:
    all_player_d2_img_copy = d2_img.copy().convert("RGBA")
    # ground truth has many copies, scale it's color down so brightness comparable
    color_alpha = int(ceil(20 * num_trajectories_for_color / float(len(trajectory_dfs))))
    ct_color = (bot_ct_color_list[0], bot_ct_color_list[1], bot_ct_color_list[2], color_alpha)
    t_color = (bot_t_color_list[0], bot_t_color_list[1], bot_t_color_list[2], color_alpha)

    num_points = 0
    with tqdm(total=len(trajectory_dfs), disable=False) as pbar:
        for trajectory_df in trajectory_dfs:
            if filter_event_type is not None:
                trajectory_df = filter_trajectory_by_key_events(filter_event_type, trajectory_df, key_areas)

            # since this was split with : rather than _, need to remove last _
            for player_place_area_columns in specific_player_place_area_columns:
                cur_player_trajectory_df = trajectory_df[trajectory_df[player_place_area_columns.alive] == 1]
                if cur_player_trajectory_df.empty:
                    continue
                player_x_coords = cur_player_trajectory_df.loc[:, player_place_area_columns.pos[0]]
                player_y_coords = cur_player_trajectory_df.loc[:, player_place_area_columns.pos[1]]
                player_canvas_x_coords, player_canvas_y_coords = \
                    convert_to_canvas_coordinates(player_x_coords, player_y_coords)
                player_xy_coords = list(zip(list(player_canvas_x_coords), list(player_canvas_y_coords)))
                num_points += len(player_x_coords)

                cur_player_d2_overlay_im = Image.new("RGBA", all_player_d2_img_copy.size, (255, 255, 255, 0))
                cur_player_d2_drw = ImageDraw.Draw(cur_player_d2_overlay_im)
                # drawing same text over and over again, so no big deal
                title_text = extra_data_from_metric_title(config.metric_cost_title, predicted)
                _, _, w, h = cur_player_d2_drw.textbbox((0, 0), title_text, font=title_font)
                cur_player_d2_drw.text(((all_player_d2_img_copy.width - w) / 2,
                                        (all_player_d2_img_copy.height * 0.1 - h) / 2),
                                       title_text, fill=(255, 255, 255, 255), font=title_font)

                ct_team = team_strs[0] in player_place_area_columns.player_id
                if ct_team:
                    if not include_ct:
                        continue
                    fill_color = ct_color
                else:
                    if not include_t:
                        continue
                    fill_color = t_color
                cur_player_d2_drw.line(xy=player_xy_coords, fill=fill_color, width=5)
                all_player_d2_img_copy.alpha_composite(cur_player_d2_overlay_im)
            pbar.update(1)

    print(f"num trajectories in plot {len(trajectory_dfs)}, alpha {color_alpha}, num points {num_points}")

    return all_player_d2_img_copy


max_distance = 1e7


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


debug_caching = True
remove_debug_cache = False
plot_ground_truth = False

def plot_trajectory_comparison_heatmaps(similarity_df: pd.DataFrame, predicted_loaded_model: LoadedModel,
                                        ground_truth_loaded_model: LoadedModel,
                                        config: ComparisonConfig, similarity_plots_path: Path):
    set_pd_print_options()
    # print(similarity_df.loc[:, [predicted_round_id_col, best_fit_ground_truth_round_id_col, metric_type_col,
    #                            dtw_cost_col, delta_distance_col, delta_time_col]])

    # plot cost, distance, and time by metric type
    metric_types = similarity_df[metric_type_col].unique().tolist()
    metric_types = [m.decode('utf-8') for m in metric_types]
    slope_metric_type = [m for m in metric_types if 'Slope' in m][0]
    relevant_similarity_df = similarity_df[(similarity_df[metric_type_col].str.decode('utf-8') == slope_metric_type) &
                                           (similarity_df[best_match_id_col] < plot_n_most_similar)]

    # same predicted will have multiple matches, start with set and convert to list
    predicted_rounds_for_comparison_heatmap: Set[RoundForComparisonHeatmap] = set()
    best_fit_ground_truth_rounds_for_comparison_heatmap: Set[RoundForComparisonHeatmap] = set()
    for _, similarity_row in relevant_similarity_df.iterrows():
        predicted_rounds_for_comparison_heatmap.add(
            RoundForComparisonHeatmap(similarity_row[predicted_trace_batch_col].decode('utf-8'),
                                  similarity_row[predicted_round_id_col],
                                  similarity_row[predicted_round_number_col])
        )
        best_fit_ground_truth_rounds_for_comparison_heatmap.add(
            RoundForComparisonHeatmap(similarity_row[best_fit_ground_truth_trace_batch_col].decode('utf-8'),
                                      similarity_row[best_fit_ground_truth_round_id_col],
                                      similarity_row[best_fit_ground_truth_round_number_col])
        )

    print('loading predicted df')
    debug_cache_path = similarity_plots_path / "predicted_trajectory_dfs.pickle"
    if debug_caching and debug_cache_path.is_file():
        with open(debug_cache_path, "rb") as pickle_file:
            predicted_trajectory_dfs = pickle.load(pickle_file)
    else:
        predicted_trajectory_dfs = select_trajectories_into_dfs(predicted_loaded_model,
                                                                list(predicted_rounds_for_comparison_heatmap))
        if debug_caching:
            with open(debug_cache_path, "wb") as pickle_file:
                pickle.dump(predicted_trajectory_dfs, pickle_file)
        elif remove_debug_cache:
            debug_cache_path.unlink(missing_ok=True)

    if plot_ground_truth:
        print('loading best fit ground truth df')
        best_fit_ground_truth_trajectory_dfs = \
            select_trajectories_into_dfs(ground_truth_loaded_model,
                                         list(best_fit_ground_truth_rounds_for_comparison_heatmap))

    #print("plotting predicted")
    #predicted_image = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, max_trajectories, True, True)
    #predicted_image.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories.png'))
    #print("plotting predicted just ct")
    #predicted_ct_image = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, max_trajectories, True, False)
    #predicted_ct_image.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_ct.png'))
    #print("plotting predicted just t")
    #predicted_t_image = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, max_trajectories, False, True)
    #predicted_t_image.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_t.png'))
    print("plotting predicted fire events")
    predicted_image = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, True, True,
                                          FilterEventType.Fire)
    predicted_image.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_fire.png'))
    print("plotting predicted kill events")
    predicted_image = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, True, True,
                                          FilterEventType.Kill)
    predicted_image.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_kill.png'))
    # first key area is cat, second is bdoors
    key_areas: KeyAreas = [AABB(Vec3(245., 1920., -50), Vec3(510., 2070., 10000))]
                           #AABB(Vec3(-1450., 2030., -10000), Vec3(-1000., 2890., 10000))]
    print("plotting predicted key area events")
    predicted_image = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, True, True,
                                          FilterEventType.KeyArea, key_areas)
    predicted_image.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_area.png'))


    if plot_ground_truth:
        ground_truth_image = plot_trajectory_dfs(best_fit_ground_truth_trajectory_dfs, config, False, True, True)

        combined_image = Image.new('RGB', (predicted_image.width + ground_truth_image.width, predicted_image.height))
        combined_image.paste(predicted_image, (0, 0))
        combined_image.paste(ground_truth_image, (predicted_image.width, 0))
        combined_image.save(similarity_plots_path / (config.metric_cost_file_name + '_similarity_trajectories' + '.png'))

    print("plotting teammate")
    plot_distance_to_teammate_enemy_from_trajectory_dfs(predicted_trajectory_dfs, config, True, similarity_plots_path)
    print("plotting enemy")
    plot_distance_to_teammate_enemy_from_trajectory_dfs(predicted_trajectory_dfs, config, False, similarity_plots_path)
