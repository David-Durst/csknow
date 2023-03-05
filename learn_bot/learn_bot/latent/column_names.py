from typing import List
from dataclasses import dataclass

from learn_bot.libs.io_transforms import ColumnTypes

max_enemies = 5
num_engagement_states = 4

row_id_column = "id"
round_id_column = "round id"
tick_id_column = "tick id"
player_id_column = "player id"


@dataclass
class EnemyColumns:
    enemy_player_id: str
    engagement_state: str
    time_since_last_visible_or_to_become_visible: str
    world_distance_to_enemy: str
    crosshair_distance_to_enemy: str
    nearest_target_enemy: str
    hit_target_enemy: str

    def to_list(self) -> list[str]:
        return [self.enemy_player_id, self.engagement_state, self.time_since_last_visible_or_to_become_visible,
                self.world_distance_to_enemy, self.crosshair_distance_to_enemy, self.nearest_target_enemy,
                self.hit_target_enemy]

    def to_input_float_list(self) -> list[str]:
        return [self.time_since_last_visible_or_to_become_visible, self.world_distance_to_enemy,
                self.crosshair_distance_to_enemy]

    def to_input_cat_list(self) -> list[str]:
        return [self.engagement_state]

    def to_input_cat_num_options(self) -> list[int]:
        return [num_engagement_states]

    def to_output_list(self) -> list[str]:
        return [self.nearest_target_enemy, self.hit_target_enemy]



base_enemy_columns: EnemyColumns = EnemyColumns(
    "enemy player id",
    "enemy engagement states",
    "time since last visible or to become visible",
    "world distance to enemy",
    "crosshair distance to enemy",
    "nearest target enemy",
    "hit target enemy"
)


def get_ith_enemy_columns(i: int) -> EnemyColumns:
    result = base_enemy_columns
    result.enemy_player_id += f" {i}"
    result.engagement_state += f" {i}"
    result.time_since_last_visible_or_to_become_visible += f" {i}"
    result.world_distance_to_enemy += f" {i}"
    result.crosshair_distance_to_enemy += f" {i}"
    result.nearest_target_enemy += f" {i}"
    result.hit_target_enemy += f" {i}"


specific_enemy_columns: list[EnemyColumns] = [get_ith_enemy_columns(i) for i in range(max_enemies)]
flat_specific_enemy_columns: list[str] = [col for cols in specific_enemy_columns for col in cols.to_list()]
flat_input_float_specific_enemy_columns: list[str] = \
    [col for cols in specific_enemy_columns for col in cols.to_input_float_list()]
flat_input_cat_specific_enemy_columns: list[str] = \
    [col for cols in specific_enemy_columns for col in cols.to_input_cat_list()]
flat_input_cat_specific_enemy_num_options: list[int] = \
    [col for cols in specific_enemy_columns for col in cols.to_input_cat_num_options()]
flat_output_float_specific_enemy_columns: list[str] = \
    [col for cols in specific_enemy_columns for col in cols.to_output_list()]


def get_simplified_column_types(float_standard_cols: List[str], categorical_cols: List[str],
                                num_cats_per_col: List[int]) -> ColumnTypes:
    return ColumnTypes(float_standard_cols, [], [], [], [], [], categorical_cols, num_cats_per_col, [], [], [])


input_column_types = get_simplified_column_types(flat_input_float_specific_enemy_columns,
                                                 flat_input_cat_specific_enemy_columns,
                                                 flat_input_cat_specific_enemy_num_options)
output_column_types = get_simplified_column_types(flat_output_float_specific_enemy_columns, [], [])
