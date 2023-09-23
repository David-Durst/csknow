import pickle
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List, Set, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.filter_trajectory_key_events import FilterEventType, \
    filter_trajectory_by_key_events, KeyAreas
from learn_bot.latent.analyze.compare_trajectories.plot_distance_to_other_player import \
    plot_distance_to_teammate_enemy_from_trajectory_dfs, extra_data_from_metric_title
from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import set_pd_print_options, \
    ComparisonConfig
from learn_bot.latent.analyze.comparison_column_names import metric_type_col, best_fit_ground_truth_trace_batch_col, predicted_trace_batch_col, best_match_id_col, predicted_round_id_col, \
    best_fit_ground_truth_round_id_col, predicted_round_number_col, best_fit_ground_truth_round_number_col
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, convert_to_canvas_coordinates, \
    bot_ct_color_list, bot_t_color_list
from learn_bot.latent.engagement.column_names import round_id_column, round_number_column
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import AABB
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


def plot_trajectory_dfs(trajectory_dfs: List[pd.DataFrame], config: ComparisonConfig, predicted: bool,
                        include_ct: bool, include_t: bool,
                        filter_event_type: Optional[FilterEventType] = None,
                        key_areas: Optional[KeyAreas] = None) -> Image.Image:
    filtered_trajectory_dfs: List[pd.DataFrame] = []
    num_points = 0
    if filter_event_type is not None:
        for round_trajectory_df in trajectory_dfs:
            # split round trajectory by filter
            cur_round_filtered_trajectory_dfs = filter_trajectory_by_key_events(filter_event_type, round_trajectory_df, key_areas)
            filtered_trajectory_dfs += cur_round_filtered_trajectory_dfs
            for df in cur_round_filtered_trajectory_dfs:
                num_points += len(df)
    else:
        filtered_trajectory_dfs = trajectory_dfs
        for df in trajectory_dfs:
            num_points += len(df)
    all_player_d2_img_copy = d2_img.copy().convert("RGBA")
    # ground truth has many copies, scale it's color down so brightness comparable
    if filter_event_type is None:
        num_points_for_color = 150000
    elif filter_event_type == FilterEventType.Fire:
        num_points_for_color = 60000
    elif filter_event_type == FilterEventType.Kill:
        num_points_for_color = 30000
    else:
        num_points_for_color = 20000
    color_alpha = int(ceil(20 * num_points_for_color / float(num_points)))
    ct_color = (bot_ct_color_list[0], bot_ct_color_list[1], bot_ct_color_list[2], color_alpha)
    t_color = (bot_t_color_list[0], bot_t_color_list[1], bot_t_color_list[2], color_alpha)

    first_title = True
    team_text = f" CT: {include_ct}, T: {include_t}"
    event_text = "" if filter_event_type is None else (" " + str(filter_event_type))

    with tqdm(total=len(filtered_trajectory_dfs), disable=False) as pbar:
        for trajectory_df in filtered_trajectory_dfs:
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

                cur_player_d2_overlay_im = Image.new("RGBA", all_player_d2_img_copy.size, (255, 255, 255, 0))
                cur_player_d2_drw = ImageDraw.Draw(cur_player_d2_overlay_im)
                if first_title:
                    title_text = extra_data_from_metric_title(config.metric_cost_title, predicted) + \
                                 event_text + team_text
                    _, _, w, h = cur_player_d2_drw.textbbox((0, 0), title_text, font=title_font)
                    cur_player_d2_drw.text(((all_player_d2_img_copy.width - w) / 2,
                                            (all_player_d2_img_copy.height * 0.1 - h) / 2),
                                           title_text, fill=(255, 255, 255, 255), font=title_font)
                    first_title = False

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


debug_caching = True
remove_debug_cache = False
plot_ground_truth = False
key_areas: KeyAreas = [AABB(Vec3(245., 2000., -50), Vec3(510., 2070., 10000))]
# AABB(Vec3(-1450., 2030., -10000), Vec3(-1000., 2890., 10000))]


class TrajectoryPlots:
    unfiltered: Image
    filtered_fire: Image
    filtered_kill: Image
    filtered_area: Image


def plot_predicted_trajectory_per_team(predicted_trajectory_dfs: List[pd.DataFrame], config: ComparisonConfig,
                                       similarity_plots_path: Path,
                                       include_ct: bool, include_t: bool) -> TrajectoryPlots:
    use_team_str = f" ct: {include_ct}, t: {include_t}"
    team_file_ending = ".png"
    if include_t and not include_ct:
        team_file_ending = "_t.png"
    elif not include_t and include_ct:
        team_file_ending = "_ct.png"

    result = TrajectoryPlots()

    print("plotting predicted" + use_team_str)
    result.unfiltered = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, include_ct, include_t)
    result.unfiltered.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories' + team_file_ending))

    print("plotting predicted fire events" + use_team_str)
    result.filtered_fire = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, include_ct, include_t,
                                               FilterEventType.Fire)
    result.filtered_fire.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_fire' +
                                                       team_file_ending))

    print("plotting predicted kill events" + use_team_str)
    result.filtered_kill = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, include_ct, include_t,
                                               FilterEventType.Kill)
    result.filtered_kill.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_kill' +
                                                       team_file_ending))

    # first key area is cat, second is bdoors
    print("plotting predicted key area events" + use_team_str)
    result.filtered_area = plot_trajectory_dfs(predicted_trajectory_dfs, config, True, include_ct, include_t,
                                               FilterEventType.KeyArea, key_areas)
    result.filtered_area.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_area' +
                                                       team_file_ending))

    return result


def concat_horizontal(ims: List[Image.Image]) -> Image.Image:
    dst = Image.new('RGB', (sum([im.width for im in ims]), ims[0].height))
    offset = 0
    for im in ims:
        dst.paste(im, (offset, 0))
        offset += im.width
    return dst


def concat_vertical(ims: List[Image.Image]) -> Image.Image:
    dst = Image.new('RGB', (ims[0].width, sum([im.height for im in ims])))
    offset = 0
    for im in ims:
        dst.paste(im, (0, offset))
        offset += im.height
    return dst


def concat_trajectory_plots(trajectory_plots: List[TrajectoryPlots], similarity_plots_path: Path,
                            config: ComparisonConfig):
    result = TrajectoryPlots()
    result.unfiltered = concat_horizontal([t.unfiltered for t in trajectory_plots])
    result.filtered_fire = concat_horizontal([t.filtered_fire for t in trajectory_plots])
    result.filtered_kill = concat_horizontal([t.filtered_kill for t in trajectory_plots])
    result.filtered_area = concat_horizontal([t.filtered_area for t in trajectory_plots])
    result_im = concat_vertical([result.unfiltered, result.filtered_fire, result.filtered_kill, result.filtered_area])
    result_im.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_all_events_teams.png'))


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

    both_teams_predicted_images = \
        plot_predicted_trajectory_per_team(predicted_trajectory_dfs, config, similarity_plots_path, True, True)
    ct_predicted_images = \
        plot_predicted_trajectory_per_team(predicted_trajectory_dfs, config, similarity_plots_path, True, False)
    t_predicted_images = \
        plot_predicted_trajectory_per_team(predicted_trajectory_dfs, config, similarity_plots_path, False, True)
    concat_trajectory_plots([both_teams_predicted_images, ct_predicted_images, t_predicted_images],
                            similarity_plots_path, config)
    quit(0)


    if plot_ground_truth:
        assert False
    #    ground_truth_image = plot_trajectory_dfs(best_fit_ground_truth_trajectory_dfs, config, False, True, True)

    #    combined_image = Image.new('RGB', (predicted_image.width + ground_truth_image.width, predicted_image.height))
    #    combined_image.paste(predicted_image, (0, 0))
    #    combined_image.paste(ground_truth_image, (predicted_image.width, 0))
    #    combined_image.save(similarity_plots_path / (config.metric_cost_file_name + '_similarity_trajectories' + '.png'))

    print("plotting teammate")
    plot_distance_to_teammate_enemy_from_trajectory_dfs(predicted_trajectory_dfs, config, True, similarity_plots_path)
    print("plotting enemy")
    plot_distance_to_teammate_enemy_from_trajectory_dfs(predicted_trajectory_dfs, config, False, similarity_plots_path)
