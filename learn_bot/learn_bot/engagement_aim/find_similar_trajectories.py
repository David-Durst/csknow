import pandas as pd
import numpy as np
from learn_bot.engagement_aim.dataset import *
from dataclasses import dataclass

from learn_bot.libs.temporal_column_names import get_temporal_field_str
from typing import Union, Optional


def compute_distance(df: Union[pd.DataFrame, pd.Series], x_col: str, y_col: str, result_col: str,
                     start_t: int, end_t: int):
    x_distance = df[get_temporal_field_str(x_col, end_t)] - df[get_temporal_field_str(x_col, start_t)]
    y_distance = df[get_temporal_field_str(y_col, end_t)] - df[get_temporal_field_str(y_col, start_t)]
    df[result_col] = (x_distance.pow(2) + y_distance.pow(2)).pow(0.5)


def compute_angular_difference(df: pd.DataFrame, selected_row_df: pd.DataFrame, x_col: str, y_col: str, result_col: str,
                               start_t: int, end_t: int):
    x_start_col = get_temporal_field_str(x_col, start_t)
    x_end_col = get_temporal_field_str(x_col, end_t)
    y_start_col = get_temporal_field_str(y_col, start_t)
    y_end_col = get_temporal_field_str(y_col, end_t)

    x_distance = df[x_end_col] - df[x_start_col]
    y_distance = df[y_end_col] - df[y_start_col]
    magnitude = (x_distance.pow(2) + y_distance.pow(2)).pow(0.5)

    selected_x_distance = selected_row_df[x_end_col] - selected_row_df[x_start_col]
    selected_y_distance = selected_row_df[y_end_col] - selected_row_df[y_start_col]
    selected_magnitude = (selected_x_distance.pow(2) + selected_y_distance.pow(2)).pow(0.5)

    df[result_col] = np.rad2deg(np.arccos(
        ((x_distance * selected_x_distance) + (y_distance * selected_y_distance)) /
        (magnitude * selected_magnitude)))


@dataclass
class SimilarityConstraints:
    max_results: int
    same_alive: bool
    same_visibility: bool
    view_relative_to_enemy_radius: float
    mouse_speed_radius: float
    mouse_direction_angular_radius: float
    base_abs_view_angle_x_col: str
    base_abs_view_angle_y_col: str
    base_relative_view_angle_x_col: str
    base_relative_view_angle_y_col: str
    speed_direction_mouse_ticks: int = 3


@dataclass
class SimilarTrajectory:
    engagement_id: int
    tick_id: int


def find_similar_trajectories(not_selected_df: pd.DataFrame, selected_df: pd.DataFrame, tick_id: int,
                              constraints: SimilarityConstraints) -> List[SimilarTrajectory]:
    result = []

    selected_row = selected_df[selected_df[tick_id_column] == tick_id].iloc[0, :].copy()
    selected_row_df = selected_row.to_frame(0).T
    derived_df = not_selected_df.copy()
    # just a base true condition to && with
    conditions = derived_df[tick_id_column] == derived_df[tick_id_column]
    if constraints.same_alive:
        conditions = conditions & (
                derived_df[cur_victim_alive_column] == selected_row[cur_victim_alive_column].item())
    if constraints.same_visibility:
        conditions = conditions & (
                derived_df[cur_victim_visible_column] == selected_row[cur_victim_visible_column].item())
    if constraints.view_relative_to_enemy_radius > 0.:
        relative_distance_col = "mouse distance to enemy"
        compute_distance(derived_df,
                         constraints.base_relative_view_angle_x_col,
                         constraints.base_relative_view_angle_y_col,
                         relative_distance_col, -1 * constraints.speed_direction_mouse_ticks, 0)
        compute_distance(selected_row_df,
                         constraints.base_relative_view_angle_x_col,
                         constraints.base_relative_view_angle_y_col,
                         relative_distance_col, -1 * constraints.speed_direction_mouse_ticks, 0)
        conditions = conditions & (derived_df[relative_distance_col] <= selected_row_df[relative_distance_col].item() +
                                   constraints.view_relative_to_enemy_radius)
    if constraints.mouse_speed_radius >= 0.:
        speed_col = "mouse speed"
        compute_distance(derived_df,
                         constraints.base_abs_view_angle_x_col,
                         constraints.base_abs_view_angle_y_col,
                         speed_col, -1 * constraints.speed_direction_mouse_ticks, 0)
        compute_distance(selected_row_df,
                         constraints.base_abs_view_angle_x_col,
                         constraints.base_abs_view_angle_y_col,
                         speed_col, -1 * constraints.speed_direction_mouse_ticks, 0)
        conditions = conditions & (derived_df[speed_col] <= selected_row_df[speed_col].item() +
                                   constraints.mouse_speed_radius)
    if constraints.mouse_direction_angular_radius >= 0.:
        angular_difference_col = "mouse direction angle difference"
        compute_angular_difference(derived_df,
                                   selected_row_df,
                                   constraints.base_abs_view_angle_x_col,
                                   constraints.base_abs_view_angle_y_col,
                                   angular_difference_col, -1 * constraints.speed_direction_mouse_ticks, 0)
        conditions = conditions & (derived_df[angular_difference_col] <= constraints.mouse_speed_radius)

    similar_df = derived_df[conditions]
    similar_engagements = similar_df.groupby([engagement_id_column]).agg({
        engagement_id_column: 'min',
        tick_id_column: 'min'
    })
    for _, row in similar_engagements.iterrows():
        result.append(SimilarTrajectory(row[engagement_id_column], row[tick_id_column]))

    return result
