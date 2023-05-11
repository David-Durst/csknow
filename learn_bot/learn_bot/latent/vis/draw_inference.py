import tkinter as tk
from tkinter import ttk, font

import pandas as pd
from PIL import Image, ImageDraw, ImageTk as itk

from learn_bot.latent.order.column_names import delta_pos_grid_num_cells, delta_pos_grid_num_cells_per_dim, \
    delta_pos_grid_cell_dim
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.mining.area_cluster import Vec3

player_width = 32

d2_top_left_x = -2476
d2_top_left_y = 3239
minimapWidth = 700
minimapHeight = 700
minimapScale = 4.4 * 1024 / minimapHeight

class VisMapCoordinate():
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
        x_index = int(cell_index / delta_pos_grid_num_cells_per_dim) - int(delta_pos_grid_num_cells_per_dim / 2)
        y_index = int(cell_index % delta_pos_grid_num_cells_per_dim) - int(delta_pos_grid_num_cells_per_dim / 2)
        new_grid_cell.coords.x -= x_index * delta_pos_grid_cell_dim
        new_grid_cell.coords.y -= y_index * delta_pos_grid_cell_dim
        return new_grid_cell

    def draw_vis(self, im_draw: ImageDraw):
        canvas_coords = self.get_canvas_coordinates()
        half_width = player_width / 2
        if not self.is_player:
            half_width = delta_pos_grid_cell_dim / 2
        x_min = canvas_coords.x - half_width
        y_min = canvas_coords.y - half_width
        x_max = canvas_coords.x + half_width
        y_max = canvas_coords.y + half_width
        if self.is_player:
            im_draw.rectangle([x_min, y_min, x_max, y_max], fill="green")
        elif self.is_prediction:
            im_draw.rectangle([x_min, y_min, x_max, y_max], fill="red")
        else:
            im_draw.rectangle([x_min, y_min, x_max, y_max], outline="blue")


def draw_all_players(data_series: pd.Series, pred_series: pd.Series, im_draw: ImageDraw):
    for player_place_area_columns in specific_player_place_area_columns:
        if data_series[player_place_area_columns.player_id] != -1:
            # draw player
            pos_coord = VisMapCoordinate(data_series[player_place_area_columns.pos[0]],
                                         data_series[player_place_area_columns.pos[1]],
                                         data_series[player_place_area_columns.pos[2]])
            pos_coord.draw_vis(im_draw)

            # draw next step and predicted next step
            max_data_prob = -1
            max_data_index = -1
            max_pred_prob = -1
            max_pred_index = -1
            for i in range(delta_pos_grid_num_cells):
                cur_data_prob = data_series[player_place_area_columns.delta_pos[i]]
                if cur_data_prob > max_data_prob:
                    max_pred_prob = cur_data_prob
                    max_data_index = i
                cur_pred_prob = pred_series[player_place_area_columns.delta_pos[i]]
                if cur_pred_prob > max_pred_prob:
                    max_pred_prob = cur_pred_prob
                    max_pred_index = i

            data_coord = pos_coord.get_grid_cell(max_data_index, False)
            data_coord.draw_vis(im_draw)
            pred_coord = pos_coord.get_grid_cell(max_pred_index, False)
            pred_coord.draw_vis(im_draw)
