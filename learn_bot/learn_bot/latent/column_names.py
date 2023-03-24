import copy
from typing import List
from dataclasses import dataclass

from learn_bot.libs.io_transforms import ColumnTypes

max_enemies = 5
num_engagement_states = 4

row_id_column = "id"
round_id_column = "round id"
tick_id_column = "tick id"
player_id_column = "player id"
pat_id_column = "player at tick id"


@dataclass
class EnemyColumns:
    enemy_player_id: str
    engagement_state: str
    time_since_last_visible_or_to_become_visible: str
    world_distance_to_enemy: str
    crosshair_distance_to_enemy: str
    nearest_target_enemy: str
    hit_target_enemy: str
    visible_1s: str
    visible_2s: str
    visible_5s: str
    visible_10s: str

    def to_list(self) -> list[str]:
        return [self.enemy_player_id, self.engagement_state, self.time_since_last_visible_or_to_become_visible,
                self.world_distance_to_enemy, self.crosshair_distance_to_enemy, self.nearest_target_enemy,
                self.hit_target_enemy, self.visible_1s, self.visible_2s, self.visible_5s, self.visible_10s]

    def to_input_float_list(self) -> list[str]:
        return [self.time_since_last_visible_or_to_become_visible, self.world_distance_to_enemy,
                self.crosshair_distance_to_enemy]

    def to_input_cat_list(self) -> list[str]:
        return [self.engagement_state]

    def to_input_cat_num_options(self) -> list[int]:
        return [num_engagement_states]

    def to_output_list(self) -> list[str]:
        return [self.nearest_target_enemy, self.hit_target_enemy,
                self.visible_1s, self.visible_2s, self.visible_5s, self.visible_10s]



base_enemy_columns: EnemyColumns = EnemyColumns(
    "enemy player id",
    "enemy engagement states",
    "time since last visible or to become visible",
    "world distance to enemy",
    "crosshair distance to enemy",
    "nearest target enemy",
    "hit target enemy",
    "visible in 1s",
    "visible in 2s",
    "visible in 5s",
    "visible in 10s",
)


def get_ith_enemy_columns(i: int) -> EnemyColumns:
    result = copy.copy(base_enemy_columns)
    result.enemy_player_id += f" {i}"
    result.engagement_state += f" {i}"
    result.time_since_last_visible_or_to_become_visible += f" {i}"
    result.world_distance_to_enemy += f" {i}"
    result.crosshair_distance_to_enemy += f" {i}"
    result.nearest_target_enemy += f" {i}"
    result.hit_target_enemy += f" {i}"
    result.visible_1s += f" {i}"
    result.visible_2s += f" {i}"
    result.visible_5s += f" {i}"
    result.visible_10s += f" {i}"
    return result


hit_engagement_column = "hit engagement"
visible_engagement_column = "visible engagement"
general_cat_columns = [hit_engagement_column, visible_engagement_column]
nearest_crosshair_500ms_column = "nearest crosshair enemy 500ms"
nearest_crosshair_1s_column = "nearest crosshair enemy 1s"
nearest_crosshair_2s_column = "nearest crosshair enemy 2s"
temporal_cat_columns = [nearest_crosshair_500ms_column, nearest_crosshair_1s_column, nearest_crosshair_2s_column]

position_offset_2s_column = "position offset 2s up to threshold"
neg_position_offset_2s_column = "neg position offset 2s up to threshold"
view_angle_offset_2s_column = "view angle offset 2s up to threshold"
neg_view_angle_offset_2s_column = "neg view angle offset 2s up to threshold"
pct_nearest_enemy_2s_columns = ["pct nearest crosshair enemy 2s " + str(i) for i in range(max_enemies + 1)]
next_tick_id_2s_column = "next tick id 2s"
next_tick_id_columns = [f"next tick id {str(2*(i+1))}s" for i in range(1,10)]


specific_enemy_columns: list[EnemyColumns] = [get_ith_enemy_columns(i) for i in range(max_enemies)]
flat_specific_enemy_columns: list[str] = [col for cols in specific_enemy_columns for col in cols.to_list()]
flat_input_float_specific_enemy_columns: list[str] = \
    [col for cols in specific_enemy_columns for col in cols.to_input_float_list()]
flat_input_cat_specific_enemy_columns: list[str] = \
    [col for cols in specific_enemy_columns for col in cols.to_input_cat_list()]
flat_input_cat_specific_enemy_num_options: list[int] = \
    [col for cols in specific_enemy_columns for col in cols.to_input_cat_num_options()]
binary_flat_output_cat_columns: list[str] = \
    [col for cols in specific_enemy_columns for col in cols.to_output_list()] + general_cat_columns
flat_output_cat_columns: list[str] = binary_flat_output_cat_columns + temporal_cat_columns
flat_output_num_options: list[int] = [2 for _ in binary_flat_output_cat_columns] + \
    [max_enemies + 1 for _ in temporal_cat_columns]
flat_output_cat_distribution_columns: list[list[str]] = \
    [[position_offset_2s_column, neg_position_offset_2s_column],
     #[view_angle_offset_2s_column, neg_view_angle_offset_2s_column],
     pct_nearest_enemy_2s_columns]


def get_simplified_column_types(float_standard_cols: List[str], categorical_cols: List[str],
                                num_cats_per_col: List[int],
                                categorical_distribution_cols: List[List[str]]) -> ColumnTypes:
    return ColumnTypes(float_standard_cols, [], [], [], [], [], categorical_cols, num_cats_per_col, [], [], [],
                       categorical_distribution_cols)


input_column_types = get_simplified_column_types(flat_input_float_specific_enemy_columns,
                                                 flat_input_cat_specific_enemy_columns,
                                                 flat_input_cat_specific_enemy_num_options, [])
#output_column_types = get_simplified_column_types([], flat_output_cat_columns, flat_output_num_options,
#                                                  flat_output_cat_distribution_columns)
output_column_types = get_simplified_column_types([], [], [],
                                                  flat_output_cat_distribution_columns)
