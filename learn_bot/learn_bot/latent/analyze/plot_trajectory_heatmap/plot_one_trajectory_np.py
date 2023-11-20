from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

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
        self.ct_buffer = np.ndarray(d2_img.size, dtype=np.intc)
        self.t_buffer = np.ndarray(d2_img.size, dtype=np.intc)

    def get_buffer(self, ct_team) -> np.ndarray:
        if ct_team:
            return self.ct_buffer
        else:
            return self.t_buffer


image_to_buffers: Dict[str, ImageBuffers]


def plot_one_trajectory_np(loaded_model: LoadedModel, id_df: pd.DataFrame, dataset: np.ndarray,
                           trajectory_filter_options: TrajectoryFilterOptions, title_str: str):
    if title_str not in image_to_buffers:
        image_to_buffers[title_str] = ImageBuffers()

    # convert this to trajectory ids for open sim trajectories
    round_ids = id_df[round_id_column].unique()
    if trajectory_filter_options.valid_round_ids is not None:
        round_ids = [r for r in round_ids if r in trajectory_filter_options.valid_round_ids]

    for round_id in round_ids:
        round_np = dataset[id_df[round_id_column] == round_id]

        for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
            ct_team = team_strs[0] in player_place_area_columns.player_id

            alive_round_np = round_np[round_np[loaded_model.model.alive_columns[player_index]] == 1]
            canvas_pos_np = convert_to_canvas_coordinates(
                alive_round_np[loaded_model.model.nested_players_pos_columns_tensor[player_index, 0:2]])
            canvas_pos_x_np = canvas_pos_np[0].astype(np.ndarray)
            canvas_pos_y_np = canvas_pos_np[1].astype(np.ndarray)

            buffer = image_to_buffers[title_str].get_buffer(ct_team)

            for i in range(len(canvas_pos_x_np)):
                buffer[canvas_pos_x_np[i], canvas_pos_y_np[i]] += 1


color_alpha = 150


def plot_one_image_one_team(title: str, ct_team: bool, team_color: List, base_img: Image.Image):
    buffer = image_to_buffers[title].get_buffer(ct_team)
    max_value = np.max(buffer)
    scaled_buffer = buffer * int(color_alpha / max_value)
    colored_buffer = scaled_buffer[:, :, np.newaxis].repeat(1, 1, 4)
    colored_buffer[:, :, 0] = team_color[0]
    colored_buffer[:, :, 1] = team_color[1]
    colored_buffer[:, :, 2] = team_color[2]
    base_img.alpha_composite(Image.fromarray(colored_buffer, 'RGBA'))


def plot_trajectories_to_image(titles: List[str], plot_teams_separately: bool, plots_path: Path):
    title_images: List[Image.Image] = []

    for title in titles:
        images_per_title: List[Image.Image] = []

        # image with everyone
        base_both_d2_img = d2_img.copy().convert("RGBA")
        plot_one_image_one_team(title, False, bot_t_color_list, base_both_d2_img)
        plot_one_image_one_team(title, True, bot_ct_color_list, base_both_d2_img)
        images_per_title.append(base_both_d2_img)

        if plot_teams_separately:
            # image with just ct
            base_ct_d2_img = d2_img.copy().convert("RGBA")
            plot_one_image_one_team(title, True, bot_ct_color_list, base_ct_d2_img)
            images_per_title.append(base_ct_d2_img)

            # image with just t
            base_t_d2_img = d2_img.copy().convert("RGBA")
            plot_one_image_one_team(title, False, bot_t_color_list, base_ct_d2_img)
            images_per_title.append(base_t_d2_img)
            title_images.append(concat_horizontal(images_per_title))
        else:
            title_images.append(images_per_title[0])

    complete_image = concat_vertical(title_images)
    complete_image.save(plots_path / 'complete_image.png')



