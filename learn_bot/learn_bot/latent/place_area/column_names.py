from enum import Enum
from typing import TypeVar

from learn_bot.latent.engagement.column_names import *
from learn_bot.latent.order.column_names import *

num_places = 26
area_grid_dim = 5
area_grid_size = area_grid_dim * area_grid_dim

c4_pos_cols = ["c4 pos x", "c4 pos y", "c4 pos z"]
float_c4_cols = [c4_distance_to_a_site_col, c4_distance_to_b_site_col] + c4_pos_cols


def get_player_pos_columns(player_index: int, team_str: str, dim_str: str) -> str:
    return "player pos " + team_str + " " + str(player_index) + " " + dim_str


def get_player_cur_place_columns(player_index: int, place_index: int, team_str: str) -> str:
    return "cur place " + str(place_index) + " " + team_str + " " + str(player_index)


def get_player_area_grid_cell_in_place_columns(player_index: int, area_grid_index: int, team_str: str) -> str:
    return "area grid cell in place " + str(area_grid_index) + " " + team_str + " " + str(player_index)


def get_player_distribution_nearest_place(player_index: int, place_index: int, team_str: str) -> str:
    return "distribution nearest place 7 to 15s " + str(place_index) + " " + team_str + " " + str(player_index)


def get_player_distribution_nearest_grid_area(player_index: int, area_grid_index: int, team_str: str) -> str:
    return "distribution nearest area grid in place 7 to 15s " + str(area_grid_index) + " " + team_str + " " + str(player_index)


class PlayerPlaceAreaColumns:
    player_id: str
    distance_to_a_site: str
    distance_to_b_site: str
    pos: list[str]
    cur_place: list[str]
    area_grid_cell_in_place: list[str]
    distribution_nearest_place: list[str]
    distribution_nearest_grid_area: list[str]

    def __init__(self, team_str: str, player_index: int):
        self.player_id = player_id_column + " " + player_team_str(team_str, player_index)
        self.distance_to_a_site = "distance to a site " + player_team_str(team_str, player_index)
        self.distance_to_b_site = "distance to b site " + player_team_str(team_str, player_index)
        self.pos = [get_player_pos_columns(player_index, team_str, dim_str) for dim_str in ["x", "y", "z"]]
        self.cur_place = []
        self.area_grid_cell_in_place = []
        self.distribution_nearest_place = []
        self.distribution_nearest_grid_area = []
        for place_index in range(num_places):
            self.cur_place \
                .append(get_player_cur_place_columns(player_index, place_index, team_str))
            self.distribution_nearest_place \
                .append(get_player_distribution_nearest_place(player_index, place_index, team_str))
        for area_grid_index in range(area_grid_size):
            self.area_grid_cell_in_place \
                .append(get_player_area_grid_cell_in_place_columns(player_index, area_grid_index, team_str))
            self.distribution_nearest_grid_area \
                .append(get_player_distribution_nearest_grid_area(player_index, area_grid_index, team_str))

    def to_list(self) -> list[str]:
        return [self.player_id, self.distance_to_a_site, self.distance_to_b_site] + \
            flatten_list([self.pos, self.cur_place,
                          self.area_grid_cell_in_place, self.distribution_nearest_place,
                          self.distribution_nearest_grid_area])

    def to_input_float_list(self) -> list[str]:
        return [self.distance_to_a_site, self.distance_to_b_site] + self.pos + \
            self.cur_place + self.area_grid_cell_in_place

    def to_output_cat_list(self, place: bool, area: bool) -> list[list[str]]:
        result = []
        if place:
            result.append(self.distribution_nearest_place)
        if area:
            result.append(self.distribution_nearest_grid_area)
        return result


specific_player_place_area_columns: list[PlayerPlaceAreaColumns] = \
    [PlayerPlaceAreaColumns(team_str, player_index) for team_str in team_strs for player_index in range(max_enemies)]
flat_input_float_place_area_columns: list[str] = \
    float_c4_cols + [col for cols in specific_player_place_area_columns for col in cols.to_input_float_list()]
flat_input_cat_place_area_columns: list[str] = [c4_status_col]
flat_input_cat_place_area_num_options: list[int] = [num_c4_status]
flat_output_cat_place_distribution_columns: list[list[str]] = \
    flatten_list([cols.to_output_cat_list(True, False) for cols in specific_player_place_area_columns])
flat_output_cat_area_distribution_columns: list[list[str]] = \
    flatten_list([cols.to_output_cat_list(False, True) for cols in specific_player_place_area_columns])

place_area_input_column_types = get_simplified_column_types(flat_input_float_place_area_columns,
                                                            flat_input_cat_place_area_columns,
                                                            flat_input_cat_place_area_num_options, [])
#output_column_types = get_simplified_column_types([], flat_output_cat_columns, flat_output_num_options,
#                                                  flat_output_cat_distribution_columns)
place_output_column_types = get_simplified_column_types([], [], [], flat_output_cat_place_distribution_columns)
area_output_column_types = get_simplified_column_types([], [], [], flat_output_cat_area_distribution_columns)