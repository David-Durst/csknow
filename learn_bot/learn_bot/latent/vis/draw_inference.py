import tkinter as tk
from tkinter import ttk, font
from typing import Tuple, Optional, List

import pandas as pd
from PIL import Image, ImageDraw, ImageTk as itk

from learn_bot.latent.order.column_names import delta_pos_grid_num_cells, delta_pos_grid_num_cells_per_dim, \
    delta_pos_grid_cell_dim, tick_id_column
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.mining.area_cluster import Vec3

player_width = 32

d2_top_left_x = -2476
d2_top_left_y = 3239
minimapWidth = 700
minimapHeight = 700
minimapScale = 4.4 * 1024 / minimapHeight

bbox_scale_factor = 2

class VisMapCoordinate():
    coords: Vec3
    is_player: bool
    is_prediction: bool

    def __init__(self, x: float, y: float, z: float, from_canvas_pixels: bool = False):
        if from_canvas_pixels:
            pctX = x / minimapWidth
            x = d2_top_left_x + minimapScale * minimapWidth * pctX
            pctY = y / minimapHeight
            y = d2_top_left_y - minimapScale * minimapHeight * pctY
        self.coords = Vec3(x, y, z)
        self.is_player = True
        self.is_prediction = False

    def get_canvas_coordinates(self) -> Vec3:
        return Vec3(
            (self.coords.x - d2_top_left_x) / minimapScale,
            (d2_top_left_y - self.coords.y) / minimapScale,
            self.coords.z
        )

    def get_grid_cell(self, cell_index: int, is_prediction: bool):
        new_grid_cell = VisMapCoordinate(self.coords.x, self.coords.y, self.coords.z, False)
        new_grid_cell.is_player = False
        new_grid_cell.is_prediction = is_prediction
        x_index = int(cell_index % delta_pos_grid_num_cells_per_dim) - int(delta_pos_grid_num_cells_per_dim / 2)
        y_index = int(cell_index / delta_pos_grid_num_cells_per_dim) - int(delta_pos_grid_num_cells_per_dim / 2)
        new_grid_cell.coords.x += x_index * delta_pos_grid_cell_dim
        new_grid_cell.coords.y += y_index * delta_pos_grid_cell_dim
        return new_grid_cell

    def draw_vis(self, im_draw: ImageDraw, use_scale: bool, custom_color: Optional[Tuple] = None):
        half_width = delta_pos_grid_cell_dim / 2
        if use_scale:
            if not self.is_player and not self.is_prediction:
                half_width *= 0.8
            elif self.is_prediction:
                half_width *= 0.6
            half_width *= bbox_scale_factor
        x_min = self.coords.x - half_width
        y_min = self.coords.y - half_width
        canvas_min = VisMapCoordinate(x_min, y_min, self.coords.z)
        canvas_min_vec = canvas_min.get_canvas_coordinates()
        x_max = self.coords.x + half_width
        y_max = self.coords.y + half_width
        canvas_max = VisMapCoordinate(x_max, y_max, self.coords.z)
        canvas_max_vec = canvas_max.get_canvas_coordinates()

        if custom_color:
            cur_color = custom_color
        elif self.is_player:
            cur_color = "green"
        elif self.is_prediction:
            cur_color = "red"
        else:
            cur_color = "blue"
        im_draw.rectangle([canvas_min_vec.x, canvas_min_vec.y, canvas_max_vec.x, canvas_max_vec.y], fill=cur_color)


def draw_all_players(data_series: pd.Series, pred_series: pd.Series, im_draw: ImageDraw, draw_max: bool,
                     players_to_draw: List[int]) -> str:
    result = ""
    for player_index in range(len(specific_player_place_area_columns)):
        if player_index not in players_to_draw:
            continue
        player_place_area_columns = specific_player_place_area_columns[player_index]
        if data_series[player_place_area_columns.player_id] != -1:
            # draw player
            pos_coord = VisMapCoordinate(data_series[player_place_area_columns.pos[0]],
                                         data_series[player_place_area_columns.pos[1]],
                                         data_series[player_place_area_columns.pos[2]])
            if draw_max:
                pos_coord.draw_vis(im_draw, True)

            # draw next step and predicted next step
            max_data_prob = -1
            max_data_index = -1
            max_pred_prob = -1
            max_pred_index = -1
            for i in range(delta_pos_grid_num_cells):
                cur_data_prob = data_series[player_place_area_columns.delta_pos[i]]
                if cur_data_prob > max_data_prob:
                    max_data_prob = cur_data_prob
                    max_data_index = i
                cur_pred_prob = pred_series[player_place_area_columns.delta_pos[i]]
                if cur_pred_prob > max_pred_prob:
                    max_pred_prob = cur_pred_prob
                    max_pred_index = i

            data_coord = pos_coord.get_grid_cell(max_data_index, False)
            pred_coord = pos_coord.get_grid_cell(max_pred_index, True)
            player_str = f'''{player_place_area_columns.player_id} pos {pos_coord.coords}, data {data_coord.coords}, pred {pred_coord.coords}'''
            result += player_str + "\n"
            print(player_str)
            if draw_max:
                data_coord.draw_vis(im_draw, True)
                pred_coord.draw_vis(im_draw, True)
            else:
                for i in range(delta_pos_grid_num_cells):
                    cur_pred_prob = pred_series[player_place_area_columns.delta_pos[i]]
                    cur_pred_coord = pos_coord.get_grid_cell(i, True)
                    cur_pred_coord.draw_vis(im_draw, False, (int(255 * (1-cur_pred_prob)), int(255 * cur_pred_prob), 0))
    return result

