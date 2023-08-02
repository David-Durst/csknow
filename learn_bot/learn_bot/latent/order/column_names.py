from enum import Enum
from typing import TypeVar
from math import isqrt

from learn_bot.latent.engagement.column_names import *


class C4Status(Enum):
    PlantedA = 0
    PlantedB = 1
    NotPlanted = 2


num_c4_status = 3
num_orders_per_site = 3
num_prior_ticks = 12
num_future_ticks = 2
num_radial_ticks = num_future_ticks + 1
num_places = 26
area_grid_dim = 5
area_grid_size = area_grid_dim * area_grid_dim

c4_status_col = "c4 status"
c4_plant_a_col = "c4 planted a"
c4_plant_b_col = "c4 planted b"
c4_not_planted_col = "c4 not planted"
c4_pos_cols = ["c4 pos x", "c4 pos y", "c4 pos z"]
c4_ticks_since_plant = ["c4 ticks since plant"]
c4_time_left_percent = ["c4 time left percent"]
c4_distance_to_a_site_col = "c4 distance to a site"
c4_distance_to_b_site_col = "c4 distance to b site"
c4_distance_to_nearest_a_order_nav_area_cols = \
    ["c4 distance to nearest a order " + str(i) + " nav area" for i in range(num_orders_per_site)]
c4_distance_to_nearest_b_order_nav_area_cols = \
    ["c4 distance to nearest b order " + str(i) + " nav area" for i in range(num_orders_per_site)]
float_c4_cols = [c4_distance_to_a_site_col, c4_distance_to_b_site_col] + \
                c4_distance_to_nearest_a_order_nav_area_cols + c4_distance_to_nearest_b_order_nav_area_cols + \
                c4_pos_cols + c4_ticks_since_plant

team_strs = ["CT", "T"]
site_strs = ["a", "b"]


class DistributionTimes(Enum):
    Time_6 = 0
    Time_15 = 1
    Time_30 = 2


def get_player_distance_order_columns(player_index: int, order_index: int, team_str: str, site_str: str) -> str:
    return "distance to nearest " + site_str + " order " + str(order_index) + " nav area " + team_str + " " + \
        str(player_index)


def get_player_nearest_order_columns(player_index: int, order_index: int, team_str: str, site_str: str) -> str:
    return "distribution nearest " + site_str + " order " + str(order_index) + " " + team_str + " " + \
        str(player_index)

def get_player_index_on_team_column(player_index: int, team_str: str) -> list[str]:
    return ["player index on team " + str(i) + " " + team_str + " " + str(player_index) for i in range(max_enemies)]

def get_player_ctteam_column(player_index: int, team_str: str) -> str:
    return "player ctTeam " + team_str + " " + str(player_index)

def get_player_alive_column(player_index: int, team_str: str) -> str:
    return "alive " + team_str + " " + str(player_index)

def get_player_view_angle_columns(player_index: int, team_str: str, dim_str: str, history_index: int = 0) -> str:
    return "player view angle " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index)) + " " + dim_str

def get_player_pos_columns(player_index: int, team_str: str, dim_str: str, history_index: int = 0) -> str:
    return "player pos " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index)) + " " + dim_str

def get_player_nearest_crosshair_distance_to_enemy_columns(player_index: int, team_str: str, history_index: int = 0) -> str:
    return "player nearest crosshair distance to enemy " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index))

def get_player_hurt_in_last_5s_columns(player_index: int, team_str: str, history_index: int = 0) -> str:
    return "player hurt in last 5s " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index))

def get_player_fire_in_last_5s_columns(player_index: int, team_str: str, history_index: int = 0) -> str:
    return "player fire in last 5s " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index))

def get_player_enemy_visible_in_last_5s_columns(player_index: int, team_str: str, history_index: int = 0) -> str:
    return "player enemy visible in last 5s " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index))

def get_player_health_columns(player_index: int, team_str: str, history_index: int = 0) -> str:
    return "player health " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index))

def get_player_armor_columns(player_index: int, team_str: str, history_index: int = 0) -> str:
    return "player armor " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index))

def get_player_aligned_pos_columns(player_index: int, team_str: str, dim_str: str, history_index: int = 0) -> str:
    return "player aligned pos " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index)) + " " + dim_str


def get_player_velocity_columns(player_index: int, team_str: str, dim_str: str, history_index: int = 0) -> str:
    return "player velocity " + team_str + " " + str(player_index) + \
        ("" if history_index == 0 else " t-" + str(history_index)) + " " + dim_str


def get_player_cur_place_columns(player_index: int, place_index: int, team_str: str) -> str:
    return "cur place " + str(place_index) + " " + team_str + " " + str(player_index)


def get_player_prior_place_columns(player_index: int, place_index: int, team_str: str, history_index: int) -> str:
    return "prior place " + str(place_index) + " " + team_str + " " + str(player_index) + " t-" + str(history_index)


def get_player_area_grid_cell_in_place_columns(player_index: int, area_grid_index: int, team_str: str) -> str:
    return "area grid cell in place " + str(area_grid_index) + " " + team_str + " " + str(player_index)


def get_delta_pos_columns(player_index: int, delta_pos_index: int, team_str: str) -> str:
    return "delta pos " + str(delta_pos_index) + " " + team_str + " " + str(player_index)


def get_radial_vel_columns(player_index: int, radial_vel_index: int, team_str: str) -> str:
    return "radial vel " + str(radial_vel_index) + " " + team_str + " " + str(player_index)

def get_future_radial_vel_columns(player_index: int, radial_vel_index: int, team_str: str, history_index: int) -> str:
    return "radial vel " + str(radial_vel_index) + " " + team_str + " " + str(player_index) + " t+" + str(history_index)

def get_player_prior_area_grid_cell_in_place_columns(player_index: int, area_grid_index: int, team_str: str,
                                                     history_index: int) -> str:
    return "prior area grid cell in place " + str(area_grid_index) + " " + team_str + " " + str(player_index) \
        + " t-" + str(history_index)


T = TypeVar('T')


def flatten_list(xss: list[list[T]]) -> list[T]:
    return [xi for xs in xss for xi in xs]


def player_team_str(team_str: str, player_index: int, uniform_space: str = False) -> str:
    if uniform_space and team_str == "T":
        return team_str + "  " + str(player_index)
    else:
        return team_str + " " + str(player_index)


class PlayerOrderColumns:
    player_id: str
    index_on_team: list[str]
    ct_team: str
    pos: list[str]
    prior_pos: list[str]
    vel: list[str]
    cur_place: list[str]
    # prior_place: list[str]
    area_grid_cell_in_place: list[str]
    # prior_area_grid_cell_in_place: list[str]
    distance_to_a_site: str
    distance_to_b_site: str
    distance_to_nearest_a_order_nav_area: list[str]
    distance_to_nearest_b_order_nav_area: list[str]
    distribution_nearest_a_order: list[str]
    #distribution_nearest_a_order_15s: list[str]
    #distribution_nearest_a_order_30s: list[str]
    distribution_nearest_b_order: list[str]
    #distribution_nearest_b_order_15s: list[str]
    #distribution_nearest_b_order_30s: list[str]

    def __init__(self, team_str: str, player_index: int):
        self.player_id = player_id_column + " " + player_team_str(team_str, player_index)
        self.index_on_team = get_player_index_on_team_column(player_index, team_str)
        self.ct_team = get_player_ctteam_column(player_index, team_str)
        self.distance_to_a_site = "distance to a site " + player_team_str(team_str, player_index)
        self.distance_to_b_site = "distance to b site " + player_team_str(team_str, player_index)
        self.pos = [get_player_pos_columns(player_index, team_str, dim_str) for dim_str in ["x", "y", "z"]]
        self.prior_pos = []
        self.vel = [get_player_velocity_columns(player_index, team_str, dim_str) for dim_str in ["x", "y", "z"]]
        self.cur_place = []
        # self.prior_place = []
        self.area_grid_cell_in_place = []
        # self.prior_area_grid_cell_in_place = []
        self.distance_to_nearest_a_order_nav_area = []
        self.distance_to_nearest_b_order_nav_area = []
        self.distribution_nearest_a_order = []
        #self.distribution_nearest_a_order_15s = []
        #self.distribution_nearest_a_order_30s = []
        self.distribution_nearest_b_order = []
        #self.distribution_nearest_b_order_15s = []
        #self.distribution_nearest_b_order_30s = []
        for order_index in range(num_orders_per_site):
            self.distance_to_nearest_a_order_nav_area \
                .append(get_player_distance_order_columns(player_index, order_index, team_str, site_strs[0]))
            self.distance_to_nearest_b_order_nav_area \
                .append(get_player_distance_order_columns(player_index, order_index, team_str, site_strs[1]))
            self.distribution_nearest_a_order \
                .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[0]))
            #self.distribution_nearest_a_order_15s \
            #    .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[0],
            #                                             DistributionTimes.Time_15))
            #self.distribution_nearest_a_order_30s \
            #    .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[0],
            #                                             DistributionTimes.Time_30))
            self.distribution_nearest_b_order \
                .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[1]))
            #self.distribution_nearest_b_order_15s \
            #    .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[1],
            #                                             DistributionTimes.Time_15))
            #self.distribution_nearest_b_order_30s \
            #    .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[1],
            #                                             DistributionTimes.Time_30))
        for place_index in range(num_places):
            self.cur_place \
                .append(get_player_cur_place_columns(player_index, place_index, team_str))
        for area_grid_index in range(area_grid_size):
            self.area_grid_cell_in_place \
                .append(get_player_area_grid_cell_in_place_columns(player_index, area_grid_index, team_str))
        for prior_tick in range(1, num_prior_ticks + 1):
            for dim_str in ["x", "y", "z"]:
                self.prior_pos.append(get_player_pos_columns(player_index, team_str, dim_str, prior_tick))
            # for place_index in range(num_places):
            #    self.prior_place \
            #        .append(get_player_prior_place_columns(player_index, place_index, team_str, prior_tick))
            # for area_grid_index in range(area_grid_size):
            #    self.prior_area_grid_cell_in_place \
            #        .append(get_player_prior_area_grid_cell_in_place_columns(player_index, area_grid_index, team_str, prior_tick))

    def to_list(self) -> list[str]:
        return [self.player_id, self.distance_to_a_site, self.distance_to_b_site] + \
            flatten_list([self.pos, self.prior_pos, self.vel, self.cur_place,  # self.prior_place,
                          self.area_grid_cell_in_place,  # self.prior_area_grid_cell_in_place,
                          self.distance_to_nearest_a_order_nav_area, self.distance_to_nearest_b_order_nav_area,
                          self.distribution_nearest_a_order, #self.distribution_nearest_a_order_15s, self.distribution_nearest_a_order_30s,
                          self.distribution_nearest_b_order])#, self.distribution_nearest_b_order_15s, self.distribution_nearest_b_order_30s])

    def to_input_float_list(self) -> list[str]:
        return [self.distance_to_a_site, self.distance_to_b_site] + \
            flatten_list([self.pos, self.prior_pos, self.vel]) + \
            flatten_list([self.distance_to_nearest_a_order_nav_area, self.distance_to_nearest_b_order_nav_area])

    def to_input_distribution_cat_list(self) -> list[list[str]]:
        return [self.cur_place, self.area_grid_cell_in_place, self.index_on_team, [self.ct_team]]

    def to_output_cat_list(self, include_6=True, include_15=True, include_30=True) -> list[list[str]]:
        result = []
        if include_6:
            result += [flatten_list([self.distribution_nearest_a_order, self.distribution_nearest_b_order])]
        #if include_15:
        #    result += [flatten_list([self.distribution_nearest_a_order_15s, self.distribution_nearest_b_order_15s])]
        #if include_30:
        #    result += [flatten_list([self.distribution_nearest_a_order_30s, self.distribution_nearest_b_order_30s])]
        return result


specific_player_order_columns: list[PlayerOrderColumns] = \
    [PlayerOrderColumns(team_str, player_index) for team_str in team_strs for player_index in range(max_enemies)]
flat_input_float_order_columns: list[str] = \
    float_c4_cols + [col for cols in specific_player_order_columns for col in cols.to_input_float_list()]
flat_input_cat_order_columns: list[str] = [c4_status_col]
flat_input_cat_order_num_options: list[int] = [num_c4_status]
flat_input_distribution_cat_order_columns: list[list[str]] = \
    flatten_list([cols.to_input_distribution_cat_list() for cols in specific_player_order_columns])
flat_output_cat_distribution_columns: list[list[str]] = \
    flatten_list([cols.to_output_cat_list(True, False, False) for cols in specific_player_order_columns])

order_input_column_types = get_simplified_column_types(flat_input_float_order_columns,
                                                       flat_input_cat_order_columns,
                                                       flat_input_cat_order_num_options,
                                                       flat_input_distribution_cat_order_columns)
# output_column_types = get_simplified_column_types([], flat_output_cat_columns, flat_output_num_options,
#                                                  flat_output_cat_distribution_columns)
order_output_column_types = get_simplified_column_types([], [], [], flat_output_cat_distribution_columns)
