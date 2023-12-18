from dataclasses import dataclass
from enum import Enum, auto
from math import log, ceil
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.draw import line_aa, line

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_and_events import title_font
from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_from_comparison import concat_horizontal, \
    concat_vertical
from learn_bot.latent.analyze.knn.plot_min_distance_rounds import game_tick_rate
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, convert_to_canvas_coordinates, \
    bot_ct_color_list, replay_ct_color_list, bot_t_color_list, replay_t_color_list
from learn_bot.latent.engagement.column_names import round_id_column, game_tick_number_column
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns


class ImageBuffers:
    ct_buffer: np.ndarray
    t_buffer: np.ndarray

    def __init__(self):
        self.ct_buffer = np.zeros(d2_img.size, dtype=np.intc)
        self.t_buffer = np.zeros(d2_img.size, dtype=np.intc)

    def get_buffer(self, ct_team) -> np.ndarray:
        if ct_team:
            return self.ct_buffer
        else:
            return self.t_buffer


spread_radius = 2
title_to_buffers: Dict[str, ImageBuffers] = {}
title_to_num_points: Dict[str, int] = {}
title_to_lifetimes: Dict[str, List[float]] = {}
title_to_speeds: Dict[str, List[float]]


def get_title_to_num_points() -> Dict[str, int]:
    return title_to_num_points


def get_title_to_lifetimes() -> Dict[str, List[float]]:
    return title_to_lifetimes


def get_title_to_speeds() -> Dict[str, List[float]]:
    return title_to_speeds


def clear_title_caches():
    global title_to_buffers, title_to_num_points, title_to_lifetimes, title_to_speeds
    title_to_buffers = {}
    title_to_num_points = {}
    title_to_lifetimes = {}
    title_to_speeds = {}


def plot_one_trajectory_dataset(loaded_model: LoadedModel, id_df: pd.DataFrame, vis_df: pd.DataFrame,
                                dataset: np.ndarray, trajectory_filter_options: TrajectoryFilterOptions, title: str):
    if title not in title_to_buffers:
        title_to_buffers[title] = ImageBuffers()
        title_to_num_points[title] = 0

    if trajectory_filter_options.trajectory_counter is None:
        trajectory_ids = id_df[round_id_column].unique()
        if trajectory_filter_options.valid_round_ids is not None:
            trajectory_ids = [r for r in trajectory_ids if r in trajectory_filter_options.valid_round_ids]
        trajectory_id_col = id_df[round_id_column]
    else:
        trajectory_ids = trajectory_filter_options.trajectory_counter.unique()
        trajectory_id_col = trajectory_filter_options.trajectory_counter
        # trajectories are already short (open sim runs for 5s), no need to restrict them further
        assert trajectory_filter_options.round_game_seconds is None

    for trajectory_id in trajectory_ids:
        trajectory_np = dataset[trajectory_id_col == trajectory_id]
        trajectory_id_df = id_df[trajectory_id_col == trajectory_id]
        trajectory_vis_df = vis_df[trajectory_id_col == trajectory_id]
        first_game_tick_number = trajectory_id_df[game_tick_number_column].iloc[0]

        # restrict to start of round if filtering based on time
        if trajectory_filter_options.round_game_seconds is not None:
            start_condition = (trajectory_id_df[game_tick_number_column] - first_game_tick_number) >= \
                              game_tick_rate * trajectory_filter_options.round_game_seconds.start
            stop_condition = (trajectory_id_df[game_tick_number_column] - first_game_tick_number) <= \
                              game_tick_rate * trajectory_filter_options.round_game_seconds.stop
            trajectory_np = trajectory_np[start_condition & stop_condition]

        # require one player in region if allowing all players when one in region
        if trajectory_filter_options.include_all_players_when_one_in_region:
            player_in_region = trajectory_np[:, 0] != trajectory_np[:, 0]
            for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
                # require offense only, as they set the positioning for everyone else
                ct_team = team_strs[0] in player_place_area_columns.player_id
                if not ct_team:
                    continue
                player_pos_x_np = trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 0]]
                player_pos_y_np = trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 1]]
                player_in_region = player_in_region | (
                    (trajectory_np[:, loaded_model.model.alive_columns[player_index]] > 0.5) &
                    (player_pos_x_np >= trajectory_filter_options.player_starts_in_region.min.x) &
                    (player_pos_x_np <= trajectory_filter_options.player_starts_in_region.max.x) &
                    (player_pos_y_np >= trajectory_filter_options.player_starts_in_region.min.y) &
                    (player_pos_y_np <= trajectory_filter_options.player_starts_in_region.max.y))
        else:
            player_in_region = trajectory_np[:, 0] == trajectory_np[:, 0]
        trajectory_np = trajectory_np[player_in_region]

        if (trajectory_filter_options.only_kill or trajectory_filter_options.only_killed or
            trajectory_filter_options.only_shots) and \
                (trajectory_filter_options.round_game_seconds is not None or
                 trajectory_filter_options.include_all_players_when_one_in_region):
            raise Exception("can't filter by game seconds or player in region and only key events like kill/killed/shot")
        if (trajectory_filter_options.compute_lifetimes or trajectory_filter_options.compute_speeds) and \
                (trajectory_filter_options.round_game_seconds is not None or
                 trajectory_filter_options.include_all_players_when_one_in_region):
            raise Exception("can't filter by game seconds or player in region and compute lifetimes or speeds")

        for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
            ct_team = team_strs[0] in player_place_area_columns.player_id

            alive_constraint = trajectory_np[:, loaded_model.model.alive_columns[player_index]] == 1
            alive_trajectory_np = trajectory_np[alive_constraint]
            alive_trajectory_vis_df = trajectory_vis_df[alive_constraint]

            # don't worry about dead players
            if len(alive_trajectory_np) == 0:
                continue

            # require start location if filtering based on starting region to only one player
            first_pos = (
                alive_trajectory_np[0, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 0]],
                alive_trajectory_np[0, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 1]]
            )
            if trajectory_filter_options.player_starts_in_region is not None and \
                    not trajectory_filter_options.include_all_players_when_one_in_region and (
                    first_pos[0] < trajectory_filter_options.player_starts_in_region.min.x or
                    first_pos[0] > trajectory_filter_options.player_starts_in_region.max.x or
                    first_pos[1] < trajectory_filter_options.player_starts_in_region.min.y or
                    first_pos[1] > trajectory_filter_options.player_starts_in_region.max.y):
                continue

            if trajectory_filter_options.only_kill + trajectory_filter_options.only_killed + \
                    trajectory_filter_options.only_shots > 1:
                raise Exception("can only filter for one type of key event at a time")
            if trajectory_filter_options.only_kill:
                event_constraint = alive_trajectory_vis_df[player_place_area_columns.player_kill_next_tick] > 0.5
            elif trajectory_filter_options.only_killed:
                event_constraint = alive_trajectory_vis_df[player_place_area_columns.player_killed_next_tick] > 0.5
            elif trajectory_filter_options.only_shots:
                event_constraint = alive_trajectory_vis_df[player_place_area_columns.player_shots_cur_tick] > 0.5
            else:
                event_constraint = None
            if event_constraint is not None:
                alive_trajectory_np = alive_trajectory_np[event_constraint]
                alive_trajectory_vis_df = alive_trajectory_vis_df[event_constraint]
            if trajectory_filter_options.compute_lifetimes and \
                    (trajectory_filter_options.only_kill or trajectory_filter_options.only_killed or
                     trajectory_filter_options.only_shots):
                raise Exception("can't filter to kill/killed/shot events and compute lifetimes")


            canvas_pos_np = convert_to_canvas_coordinates(
                alive_trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 0]],
                alive_trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 1]])
            canvas_pos_x_np = canvas_pos_np[0].astype(np.intc)
            canvas_pos_y_np = canvas_pos_np[1].astype(np.intc)
            canvas_pos_xy = list(zip(list(canvas_pos_x_np), list(canvas_pos_y_np)))

            buffer = title_to_buffers[title].get_buffer(ct_team)
            if trajectory_filter_options.only_kill or trajectory_filter_options.only_killed:
                for pos_xy in canvas_pos_xy:
                    buffer[pos_xy[0], pos_xy[1]] += 1
            else:
                cur_player_d2_img = Image.new("L", d2_img.size, color=0)
                cur_player_d2_drw = ImageDraw.Draw(cur_player_d2_img)
                cur_player_d2_drw.line(xy=canvas_pos_xy, fill=1, width=5)
                buffer += np.asarray(cur_player_d2_img)
            title_to_num_points[title] += len(alive_trajectory_np)
            if trajectory_filter_options.compute_lifetimes:
                if title not in title_to_lifetimes:
                    title_to_lifetimes[title] = []
                lifetime_in_game_ticks = trajectory_id_df[game_tick_number_column].iloc[len(alive_trajectory_np) - 1] - \
                    first_game_tick_number
                title_to_lifetimes[title].append(lifetime_in_game_ticks / game_tick_rate)
            if trajectory_filter_options.compute_speeds:
                if title not in title_to_speeds:
                    title_to_speeds[title] = []
                speeds = (alive_trajectory_vis_df[player_place_area_columns.vel] ** 2.).sum(axis=1) ** 0.5
                title_to_speeds[title] += speeds.tolist()


scale_factor = 0
max_value = 0


def plot_one_image_one_team(title: str, ct_team: bool, team_color: List, saturated_team_color: List,
                            base_img: Image.Image, custom_buffer: Optional[np.ndarray] = None):
    if custom_buffer is None:
        buffer = title_to_buffers[title].get_buffer(ct_team)
    else:
        buffer = custom_buffer
    color_buffer = buffer[:, :, np.newaxis].repeat(4, axis=2)
    color_buffer[:, :, 0] = team_color[0]
    color_buffer[:, :, 1] = team_color[1]
    color_buffer[:, :, 2] = team_color[2]

    # if saturate, then move to darker color to indicate
    saturated_color_buffer_entries = color_buffer[:, :, 3] >= 255
    if np.sum(saturated_color_buffer_entries) > 0:
        percent_saturated = color_buffer[saturated_color_buffer_entries][:, 3] / max_value
        full_alpha_team_color = np.asarray(team_color)
        full_alpha_team_color[3] = 255
        full_alpha_saturated_team_color = np.asarray(saturated_team_color)
        full_alpha_saturated_team_color[3] = 255
        color_buffer[saturated_color_buffer_entries] = \
            (percent_saturated[:, np.newaxis].repeat(4, axis=1) *
             full_alpha_saturated_team_color[np.newaxis, :].repeat(np.sum(saturated_color_buffer_entries), axis=0)) + \
            ((1 - percent_saturated[:, np.newaxis].repeat(4, axis=1)) *
             full_alpha_team_color[np.newaxis, :].repeat(np.sum(saturated_color_buffer_entries), axis=0))
    uint8_color_buffer = np.uint8(color_buffer)
    base_img.alpha_composite(Image.fromarray(uint8_color_buffer, 'RGBA'))

    title_drw = ImageDraw.Draw(base_img)
    if custom_buffer is None:
        title_text = title + f", \n Num Points Both Teams {title_to_num_points[title]} Scale Factor {scale_factor}"
    else:
        title_text = title
    _, _, w, h = title_drw.textbbox((0, 0), title_text, font=title_font)
    title_drw.text(((d2_img.width - w) / 2, (d2_img.height * 0.1 - h) / 2),
                   title_text, fill=(255, 255, 255, 255), font=title_font)


def scale_buffers_by_points(titles: List[str]):
    global scale_factor, max_value

    max_points_per_title = 0
    for title in titles:
        max_points_per_title = max(max_points_per_title, title_to_num_points[title])
    if max_points_per_title > 1e6:
        scale_factor = 8
    elif max_points_per_title > 5e5:
        scale_factor = 11
    elif max_points_per_title > 1e5:
        scale_factor = 14
    elif max_points_per_title > 5e4:
        scale_factor = 19
    elif max_points_per_title > 1e4:
        scale_factor = 25
    else:
        scale_factor = 30
    #scale_factor = int(25. / log(2.2 + max_points_per_title / 1300, 10))
    # compute scaling factor for points
    #max_99_percentile = -1
    #for title in titles:
    #    ct_buffer = title_to_buffers[title].get_buffer(True)
    #    max_99_percentile = max(max_99_percentile, np.percentile(ct_buffer, 99))
    #    #print(f'ct_buffer percentiles: f{np.percentile(ct_buffer, [50, 90, 95, 99, 99.9, 99.99, 99.999, 99.9999])}')
    #    t_buffer = title_to_buffers[title].get_buffer(False)
    #    max_99_percentile = max(max_99_percentile, np.percentile(t_buffer, 99))
    #    #print(f't_buffer percentiles: f{np.percentile(ct_buffer, [50, 90, 95, 99, 99.9, 99.99, 99.999, 99.9999])}')
    #scale_factor = int(ceil(255 / max_99_percentile))

    # compute max value for color overflow
    max_value = -1
    for title in titles:
        ct_buffer = title_to_buffers[title].get_buffer(True)
        ct_buffer *= scale_factor
        max_value = max(max_value, np.max(ct_buffer))
        t_buffer = title_to_buffers[title].get_buffer(False)
        t_buffer *= scale_factor
        max_value = max(max_value, np.max(t_buffer))
        #print(f"{title} ct max value {np.max(ct_buffer)}, argmax {np.unravel_index(ct_buffer.argmax(), ct_buffer.shape)}")
        #print(f"{title} t max value {np.max(t_buffer)}, argmax {np.unravel_index(t_buffer.argmax(), t_buffer.shape)}")


saturated_ct_color_list = [19, 2, 178, 0]
saturated_t_color_list = [178, 69, 2, 0]


def plot_trajectories_to_image(titles: List[str], plot_teams_separately: bool, plots_path: Path,
                               trajectory_filter_options: TrajectoryFilterOptions):
    title_images: List[Image.Image] = []

    scale_buffers_by_points(titles)
    #print(f"max pixel value after scaling before clamp to 255 {max_value}")

    for title in titles:
        images_per_title: List[Image.Image] = []

        # image with everyone
        base_both_d2_img = d2_img.copy().convert("RGBA")
        plot_one_image_one_team(title, True, bot_ct_color_list, saturated_ct_color_list, base_both_d2_img)
        plot_one_image_one_team(title, False, bot_t_color_list, saturated_t_color_list, base_both_d2_img)
        images_per_title.append(base_both_d2_img)

        if plot_teams_separately:
            # image with just ct
            base_ct_d2_img = d2_img.copy().convert("RGBA")
            plot_one_image_one_team(title, True, bot_ct_color_list, saturated_ct_color_list, base_ct_d2_img)
            images_per_title.append(base_ct_d2_img)

            # image with just t
            base_t_d2_img = d2_img.copy().convert("RGBA")
            plot_one_image_one_team(title, False, bot_t_color_list, saturated_t_color_list, base_t_d2_img)
            images_per_title.append(base_t_d2_img)
            title_images.append(concat_horizontal(images_per_title))
        else:
            title_images.append(images_per_title[0])

    complete_image = concat_vertical(title_images)
    complete_image.save(plots_path / (str(trajectory_filter_options) + '.png'))


base_positive_color_list = [31, 210, 93, 0]
saturated_positive_color_list = [17, 162, 144, 0]
base_negative_color_list = [242, 48, 48, 0]
saturated_negative_color_list = [162, 7, 146, 0]


def plot_diff_one_image_one_team(title0: str, title1: str, ct_team: bool) -> Image.Image:
    buffer0 = title_to_buffers[title0].get_buffer(ct_team)
    buffer1 = title_to_buffers[title1].get_buffer(ct_team)
    delta_buffer = buffer0 - buffer1

    # split delta_buffer into two sub buffers (one positive, one negative), plot them independently
    positive_delta_buffer = np.where(delta_buffer >= 0, delta_buffer, 0)
    negative_delta_buffer = np.where(delta_buffer < 0, -1 * delta_buffer, 0)

    base_green_img = Image.new("RGBA", d2_img.size, (0, 0, 0, 255))
    ct_team_str = 'CT' if ct_team else 'T'
    plot_one_image_one_team(f"{title0} (Green) vs {title1} (Red) {ct_team_str}", ct_team, base_positive_color_list,
                            saturated_positive_color_list, base_green_img, positive_delta_buffer)
    green_np = np.asarray(base_green_img)
    base_red_img = Image.new("RGBA", d2_img.size, (0, 0, 0, 255))
    plot_one_image_one_team(f"{title0} (Green) vs {title1} (Red) {ct_team_str}", ct_team, base_negative_color_list,
                            saturated_negative_color_list, base_red_img, negative_delta_buffer)
    red_np = np.asarray(base_red_img)

    combined_np = red_np + green_np
    #base_img = d2_img.copy().convert("RGBA")
    #base_img.alpha_composite(Image.fromarray(combined_np, 'RGBA'))
    #return base_img
    return Image.fromarray(combined_np, 'RGBA')


def plot_trajectory_diffs_to_image(titles: List[str], diff_indices: List[int], plots_path: Path,
                                   trajectory_filter_options: TrajectoryFilterOptions):
    # assuming already called plot_trajectories_to_image, so already did scaling
    title_images: List[Image.Image] = []

    for i, title in enumerate(titles[1:]):
        images_per_title: List[Image.Image] = []

        images_per_title.append(plot_diff_one_image_one_team(title, titles[diff_indices[i]], True))
        images_per_title.append(plot_diff_one_image_one_team(title, titles[diff_indices[i]], False))

        title_images.append(concat_horizontal(images_per_title))

    complete_image = concat_vertical(title_images)
    complete_image.save(plots_path / 'diff' / (str(trajectory_filter_options) + '.png'))
