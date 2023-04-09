from enum import Enum
from typing import TypeVar

from learn_bot.latent.engagement.column_names import *


class C4Status(Enum):
    PlantedA = 0
    PlantedB = 1
    NotPlanted = 2


num_c4_status = 3
num_orders_per_site = 3

c4_status_col = "c4 status"
c4_distance_to_a_site_col = "c4 distance to a site"
c4_distance_to_b_site_col = "c4 distance to b site"
c4_distance_to_nearest_a_order_nav_area_cols = \
    ["c4 distance to nearest a order " + str(i) + " nav area" for i in range(num_orders_per_site)]
c4_distance_to_nearest_b_order_nav_area_cols = \
    ["c4 distance to nearest b order " + str(i) + " nav area" for i in range(num_orders_per_site)]
float_c4_cols = [c4_distance_to_a_site_col, c4_distance_to_b_site_col] + \
          c4_distance_to_nearest_a_order_nav_area_cols + c4_distance_to_nearest_b_order_nav_area_cols

team_strs = ["CT", "T"]
site_strs = ["a", "b"]


class DistributionTimes(Enum):
    Time_15 = 0
    Time_30 = 1


def get_player_distance_order_columns(player_index: int, order_index: int, team_str: str, site_str: str) -> str:
    return "distance to nearest " + site_str + " order " + str(order_index) + " nav area " + team_str + " " + \
        str(player_index)


def get_player_nearest_order_columns(player_index: int, order_index: int, team_str: str, site_str: str,
                                     distribution_time: DistributionTimes) -> str:
    if distribution_time == DistributionTimes.Time_30:
        time_str = " 30s "
    else:
        time_str = " 15s "
    return "distribution nearest " + site_str + " order " + str(order_index) + time_str + team_str + " " + \
        str(player_index)


T = TypeVar('T')


def flatten_list(xss: list[list[T]]) -> list[T]:
    return [xi for xs in xss for xi in xs]

def player_team_str(team_str: str, player_index: int) -> str:
    return team_str + " " + str(player_index)

class PlayerOrderColumns:
    player_id: str
    distance_to_a_site: str
    distance_to_b_site: str
    distance_to_nearest_a_order_nav_area: list[str]
    distance_to_nearest_b_order_nav_area: list[str]
    distribution_nearest_a_order_15s: list[str]
    distribution_nearest_a_order_30s: list[str]
    distribution_nearest_b_order_15s: list[str]
    distribution_nearest_b_order_30s: list[str]

    def __init__(self, team_str: str, player_index: int):
        self.player_id = player_id_column + " " + player_team_str(team_str, player_index)
        self.distance_to_a_site = "distance to a site " + player_team_str(team_str, player_index)
        self.distance_to_b_site = "distance to b site " + player_team_str(team_str, player_index)
        self.distance_to_nearest_a_order_nav_area = []
        self.distance_to_nearest_b_order_nav_area = []
        self.distribution_nearest_a_order_15s = []
        self.distribution_nearest_a_order_30s = []
        self.distribution_nearest_b_order_15s = []
        self.distribution_nearest_b_order_30s = []
        for order_index in range(num_orders_per_site):
            self.distance_to_nearest_a_order_nav_area \
                .append(get_player_distance_order_columns(player_index, order_index, team_str, site_strs[0]))
            self.distance_to_nearest_b_order_nav_area \
                .append(get_player_distance_order_columns(player_index, order_index, team_str, site_strs[1]))
            self.distribution_nearest_a_order_15s \
                .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[0],
                                                         DistributionTimes.Time_15))
            self.distribution_nearest_a_order_30s \
                .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[0],
                                                         DistributionTimes.Time_30))
            self.distribution_nearest_b_order_15s \
                .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[1],
                                                         DistributionTimes.Time_15))
            self.distribution_nearest_b_order_30s \
                .append(get_player_nearest_order_columns(player_index, order_index, team_str, site_strs[1],
                                                         DistributionTimes.Time_30))

    def to_list(self) -> list[str]:
        return [self.player_id, self.distance_to_a_site, self.distance_to_b_site] + \
            flatten_list([self.distance_to_nearest_a_order_nav_area, self.distance_to_nearest_b_order_nav_area,
                          self.distribution_nearest_a_order_15s, self.distribution_nearest_a_order_30s,
                          self.distribution_nearest_b_order_15s, self.distribution_nearest_b_order_30s])

    def to_input_float_list(self) -> list[str]:
        return [self.distance_to_a_site, self.distance_to_b_site] + \
            flatten_list([self.distance_to_nearest_a_order_nav_area, self.distance_to_nearest_b_order_nav_area])

    def to_input_cat_list(self) -> list[str]:
        return [self.engagement_state]

    def to_input_cat_num_options(self) -> list[int]:
        return [num_engagement_states]

    def to_output_cat_list(self, include_15=True, include_30=True) -> list[list[str]]:
        result = []
        if include_15:
            result += [flatten_list([self.distribution_nearest_a_order_15s, self.distribution_nearest_b_order_15s])]
        if include_30:
            result += [flatten_list([self.distribution_nearest_a_order_30s, self.distribution_nearest_b_order_30s])]
        return result


specific_player_order_columns: list[PlayerOrderColumns] = \
    [PlayerOrderColumns(team_str, player_index) for team_str in team_strs for player_index in range(max_enemies)]
flat_input_float_order_columns: list[str] = \
    float_c4_cols + [col for cols in specific_player_order_columns for col in cols.to_input_float_list()]
flat_input_cat_order_columns: list[str] = [c4_status_col]
flat_input_cat_order_num_options: list[int] = [num_c4_status]
flat_output_cat_distribution_columns: list[list[str]] = \
    flatten_list([cols.to_output_cat_list(False, True) for cols in specific_player_order_columns])

order_input_column_types = get_simplified_column_types(flat_input_float_order_columns,
                                                      flat_input_cat_order_columns,
                                                      flat_input_cat_order_num_options, [])
#output_column_types = get_simplified_column_types([], flat_output_cat_columns, flat_output_num_options,
#                                                  flat_output_cat_distribution_columns)
order_output_column_types = get_simplified_column_types([], [], [], flat_output_cat_distribution_columns)
