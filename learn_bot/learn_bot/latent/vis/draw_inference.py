import tkinter as tk
from tkinter import ttk, font
from typing import Tuple, Optional, List, Dict
from math import sqrt, pow, cos, sin, radians

import pandas as pd
from PIL import Image, ImageDraw, ImageTk as itk

from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import delta_pos_grid_num_cells, delta_pos_grid_num_cells_per_xy_dim, \
    delta_pos_grid_cell_dim, delta_pos_grid_num_xy_cells_per_z_change
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, num_radial_bins, \
    num_radial_bins_per_z_axis, StatureOptions, direction_angle_range
from learn_bot.latent.transformer_nested_hidden_latent_model import stature_to_speed_list
from learn_bot.mining.area_cluster import Vec3

player_width = 32

d2_top_left_x = -2476
d2_top_left_y = 3239
minimapWidth = 1300
minimapHeight = 1300
minimapScale = 4.4 * 1024 / minimapHeight

bbox_scale_factor = 2
dot_radius = 0.5

class VisMapCoordinate():
    coords: Vec3
    is_player: bool
    is_prediction: bool
    z_index: int

    def __init__(self, x: float, y: float, z: float, from_canvas_pixels: bool = False):
        if from_canvas_pixels:
            pctX = x / minimapWidth
            x = d2_top_left_x + minimapScale * minimapWidth * pctX
            pctY = y / minimapHeight
            y = d2_top_left_y - minimapScale * minimapHeight * pctY
        self.coords = Vec3(x, y, z)
        self.is_player = True
        self.is_prediction = False
        self.z_index = -1

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
        new_grid_cell.z_index = int(cell_index / delta_pos_grid_num_xy_cells_per_z_change)
        xy_cell_index = cell_index % delta_pos_grid_num_xy_cells_per_z_change
        x_index = int(xy_cell_index % delta_pos_grid_num_cells_per_xy_dim) - int(delta_pos_grid_num_cells_per_xy_dim / 2)
        y_index = int(xy_cell_index / delta_pos_grid_num_cells_per_xy_dim) - int(delta_pos_grid_num_cells_per_xy_dim / 2)
        new_grid_cell.coords.x += x_index * delta_pos_grid_cell_dim
        new_grid_cell.coords.y += y_index * delta_pos_grid_cell_dim
        return new_grid_cell

    def get_radial_cell(self, radial_index: int, is_prediction: bool):
        new_radial_cell = VisMapCoordinate(self.coords.x, self.coords.y, self.coords.z, False)
        new_radial_cell.is_player = False
        new_radial_cell.is_prediction = is_prediction
        not_moving = radial_index == 0
        moving_radial_index = radial_index - 1
        new_radial_cell.z_index = int(moving_radial_index / num_radial_bins_per_z_axis)
        dir_stature_radial_index = moving_radial_index % num_radial_bins_per_z_axis
        stature_index = int(dir_stature_radial_index % StatureOptions.NUM_STATURE_OPTIONS.value)
        speed = stature_to_speed_list[stature_index]
        dir_index = int(dir_stature_radial_index / StatureOptions.NUM_STATURE_OPTIONS.value)
        dir_degrees = dir_index * direction_angle_range
        # divide by 2 for speed as looking at half second
        if not not_moving:
            new_radial_cell.coords.x += cos(radians(dir_degrees)) * speed / 2.
            new_radial_cell.coords.y += sin(radians(dir_degrees)) * speed / 2.
        else:
            # for vis purposes, not moving has no change in z axis
            new_radial_cell.z_index = 1
        return new_radial_cell

    def draw_vis(self, im_draw: ImageDraw, use_scale: bool, custom_color: Optional[Tuple] = None, rectangle = True):
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

        # flip as csgo has different coordinates from pillow
        if rectangle:
            im_draw.rectangle([canvas_min_vec.x, canvas_max_vec.y, canvas_max_vec.x, canvas_min_vec.y], fill=cur_color)
        else:
            im_draw.ellipse([canvas_min_vec.x, canvas_max_vec.y, canvas_max_vec.x, canvas_min_vec.y], fill=cur_color)


def draw_all_players(data_series: pd.Series, pred_series: Optional[pd.Series], im_draw: ImageDraw, draw_max: bool,
                     players_to_draw: List[int], draw_only_pos: bool = False, player_to_color: Dict[int, Tuple] = {},
                     rectangle = True, radial_vel_time_step: int = 0) -> str:
    result = ""
    # colors track by number of players drawn
    for player_index in range(len(specific_player_place_area_columns)):
        player_place_area_columns = specific_player_place_area_columns[player_index]
        if data_series[player_place_area_columns.player_id] != -1:
            if player_index not in players_to_draw:
                continue

            # draw player
            pos_coord = VisMapCoordinate(data_series[player_place_area_columns.pos[0]],
                                         data_series[player_place_area_columns.pos[1]],
                                         data_series[player_place_area_columns.pos[2]])
            if draw_only_pos:
                custom_color = None
                if player_index in player_to_color:
                    custom_color = player_to_color[player_index]
                pos_coord.draw_vis(im_draw, True, custom_color=custom_color, rectangle=rectangle)
                player_str = f'''{player_place_area_columns.player_id} pos {pos_coord.coords}'''
                result += player_str + "\n"
                continue
            elif draw_max:
                pos_coord.draw_vis(im_draw, True)

            # draw next step and predicted next step
            max_data_prob = -1
            max_data_index = -1
            max_pred_prob = -1
            max_pred_index = -1
            for i in range(num_radial_bins):
                radial_vel_column = player_place_area_columns.radial_vel[i]
                if radial_vel_time_step > 0:
                    radial_vel_column = player_place_area_columns.future_radial_vel[radial_vel_time_step-1][i]
                cur_data_prob = data_series[radial_vel_column]
                if cur_data_prob > max_data_prob:
                    max_data_prob = cur_data_prob
                    max_data_index = i
                # TODO: use radial_vel_column
                cur_pred_prob = pred_series[player_place_area_columns.radial_vel[i]]
                if cur_pred_prob > max_pred_prob:
                    max_pred_prob = cur_pred_prob
                    max_pred_index = i

            data_coord = pos_coord.get_radial_cell(max_data_index, False)
            pred_coord = pos_coord.get_radial_cell(max_pred_index, True)
            vel_per_player = Vec3(data_series[player_place_area_columns.vel[0]],
                                  data_series[player_place_area_columns.vel[1]],
                                  data_series[player_place_area_columns.vel[2]])
            decrease_distance_to_c4_5s = data_series[player_place_area_columns.decrease_distance_to_c4_5s]
            decrease_distance_to_c4_10s = data_series[player_place_area_columns.decrease_distance_to_c4_10s]
            decrease_distance_to_c4_20s = data_series[player_place_area_columns.decrease_distance_to_c4_20s]
            player_str = f"{player_place_area_columns.player_id} pos {pos_coord.coords}, " \
                         f"data {data_coord.coords} {data_coord.z_index}, " \
                         f"pred {pred_coord.coords} {pred_coord.z_index}, " \
                         f"vel {vel_per_player}, radial index {max_data_index}," \
                         f"decrease distance to c4 5s {decrease_distance_to_c4_5s} 10s {decrease_distance_to_c4_10s} 20s {decrease_distance_to_c4_20s}"
            result += player_str + "\n"
            #print(player_str)
            if draw_max:
                # filter out ends of players lives when there is no pred prob
                if max_data_prob > 0.5:
                    data_coord.draw_vis(im_draw, True)
                pred_coord.draw_vis(im_draw, True)
            else:
                xy_coord_to_sum_prob: Dict[Tuple[float, float], float] = {}
                xy_coord_to_max_prob: Dict[Tuple[float, float], float] = {}
                xy_coord_to_sum_coord: Dict[Tuple[float, float], VisMapCoordinate] = {}
                xy_coord_to_max_prob_z_index: Dict[Tuple[float, float], int] = {}
                for i in range(num_radial_bins):
                    cur_pred_prob = pred_series[player_place_area_columns.radial_vel[i]]
                    cur_pred_coord = pos_coord.get_radial_cell(i, True)
                    xy_coord = cur_pred_coord.coords.x, cur_pred_coord.coords.y
                    if xy_coord not in xy_coord_to_sum_prob:
                        xy_coord_to_sum_prob[xy_coord] = 0
                        xy_coord_to_max_prob[xy_coord] = -1.
                        xy_coord_to_sum_coord[xy_coord] = cur_pred_coord
                        xy_coord_to_max_prob_z_index[xy_coord] = cur_pred_coord.z_index
                    xy_coord_to_sum_prob[xy_coord] += cur_pred_prob
                    if cur_pred_prob > xy_coord_to_max_prob[xy_coord]:
                        xy_coord_to_max_prob[xy_coord] = cur_pred_prob
                        xy_coord_to_max_prob_z_index[xy_coord] = cur_pred_coord.z_index
                for xy_coord, prob in xy_coord_to_sum_prob.items():
                    if prob < 0.05:
                        color = (255, 0, 0, 100)
                    else:
                        clamped_prob = max(0.5, min(1., prob))
                        if xy_coord_to_max_prob_z_index[xy_coord] == 0:
                            color = (0, 0, int(255 * clamped_prob), 255)
                        if xy_coord_to_max_prob_z_index[xy_coord] == 1:
                            color = (0, int(255 * clamped_prob), 0, 255)
                        elif xy_coord_to_max_prob_z_index[xy_coord] == 2:
                            color = (0, int(255 * clamped_prob), int(255 * clamped_prob), 255)
                    xy_coord_to_sum_coord[xy_coord].draw_vis(im_draw, False, color)
    return result


def draw_player_connection_lines(src_data_series: pd.Series, tgt_data_series, im_draw: ImageDraw,
                                src_to_tgt_player_index: Dict[int, int], players_to_draw: List[int],
                                src_player_to_color: Dict[int, Tuple]):
    # colors track by number of players drawn
    for player_index in range(len(specific_player_place_area_columns)):
        player_place_area_columns = specific_player_place_area_columns[player_index]
        if src_data_series[player_place_area_columns.player_id] != -1:
            if player_index not in players_to_draw:
                continue

            src_pos_coord = VisMapCoordinate(src_data_series[player_place_area_columns.pos[0]],
                                             src_data_series[player_place_area_columns.pos[1]],
                                             src_data_series[player_place_area_columns.pos[2]])

            if player_index not in src_to_tgt_player_index:
                continue
            tgt_player_index = src_to_tgt_player_index[player_index]
            tgt_player_place_area_columns = specific_player_place_area_columns[tgt_player_index]
            if tgt_data_series[tgt_player_place_area_columns.player_id] == -1:
                continue
            tgt_pos_coord = VisMapCoordinate(tgt_data_series[tgt_player_place_area_columns.pos[0]],
                                             tgt_data_series[tgt_player_place_area_columns.pos[1]],
                                             tgt_data_series[tgt_player_place_area_columns.pos[2]])
            distance = sqrt(pow(src_pos_coord.coords.x - tgt_pos_coord.coords.x, 2) +
                            pow(src_pos_coord.coords.y - tgt_pos_coord.coords.y, 2))

            if distance > 200.:
                src_canvas_coords = src_pos_coord.get_canvas_coordinates()
                tgt_canvas_coords = tgt_pos_coord.get_canvas_coordinates()
                im_draw.line([src_canvas_coords.x, src_canvas_coords.y, tgt_canvas_coords.x, tgt_canvas_coords.y],
                             fill=src_player_to_color[player_index], width=5)

