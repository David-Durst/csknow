from dataclasses import dataclass
from enum import Enum, auto
from math import log
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.draw import line_aa, line

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_and_events import title_font
from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_from_comparison import concat_horizontal, \
    concat_vertical
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, convert_to_canvas_coordinates, \
    bot_ct_color_list, replay_ct_color_list, bot_t_color_list, replay_t_color_list
from learn_bot.latent.engagement.column_names import round_id_column
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


image_to_buffers: Dict[str, ImageBuffers] = {}
image_to_num_points: Dict[str, int] = {}
spread_radius = 2


def plot_one_trajectory_dataset(loaded_model: LoadedModel, id_df: pd.DataFrame, dataset: np.ndarray,
                                trajectory_filter_options: TrajectoryFilterOptions, title_str: str):
    if title_str not in image_to_buffers:
        image_to_buffers[title_str] = ImageBuffers()
        image_to_num_points[title_str] = 0

    # convert this to trajectory ids for open sim trajectories
    round_ids = id_df[round_id_column].unique()
    if trajectory_filter_options.valid_round_ids is not None:
        round_ids = [r for r in round_ids if r in trajectory_filter_options.valid_round_ids]

    for round_id in round_ids:
        round_np = dataset[id_df[round_id_column] == round_id]

        for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
            ct_team = team_strs[0] in player_place_area_columns.player_id

            alive_round_np = round_np[round_np[:, loaded_model.model.alive_columns[player_index]] == 1]
            canvas_pos_np = convert_to_canvas_coordinates(
                alive_round_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 0]],
                alive_round_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 1]])
            canvas_pos_x_np = canvas_pos_np[0].astype(np.intc)
            canvas_pos_y_np = canvas_pos_np[1].astype(np.intc)
            canvas_pos_xy = list(zip(list(canvas_pos_x_np), list(canvas_pos_y_np)))

            buffer = image_to_buffers[title_str].get_buffer(ct_team)
            cur_player_d2_img = Image.new("L", d2_img.size, color=0)
            cur_player_d2_drw = ImageDraw.Draw(cur_player_d2_img)
            cur_player_d2_drw.line(xy=canvas_pos_xy, fill=1, width=5)
            buffer += np.asarray(cur_player_d2_img)
            image_to_num_points[title_str] += len(alive_round_np)


scale_factor = 0


def plot_one_image_one_team(title: str, ct_team: bool, team_color: List, saturated_team_color: List,
                            base_img: Image.Image):
    buffer = image_to_buffers[title].get_buffer(ct_team)
    max_value = np.max(buffer)
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
    title_text = title + f", \n Num Points Both Teams {image_to_num_points[title]} Scale Factor {scale_factor}"
    _, _, w, h = title_drw.textbbox((0, 0), title_text, font=title_font)
    title_drw.text(((d2_img.width - w) / 2, (d2_img.height * 0.1 - h) / 2),
                   title_text, fill=(255, 255, 255, 255), font=title_font)


def scale_buffers_by_points(titles: List[str]):
    global scale_factor
    max_points_per_title = 0
    for title in titles:
        max_points_per_title = max(max_points_per_title, image_to_num_points[title])
    scale_factor = int(25. / log(2.2 + max_points_per_title / 1300, 10))
    ct_buffer = image_to_buffers[title].get_buffer(True)
    ct_buffer *= scale_factor
    t_buffer = image_to_buffers[title].get_buffer(False)
    t_buffer *= scale_factor


saturated_ct_color_list = [19, 2, 178, 0]
saturated_t_color_list = [178, 69, 2, 0]


def plot_trajectories_to_image(titles: List[str], plot_teams_separately: bool, plots_path: Path):
    title_images: List[Image.Image] = []

    scale_buffers_by_points(titles)

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
    complete_image.save(plots_path / 'complete_image.png')



