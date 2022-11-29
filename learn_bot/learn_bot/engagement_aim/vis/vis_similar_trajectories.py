import pandas as pd
import numpy as np
from learn_bot.engagement_aim.dataset import *
from dataclasses import dataclass
import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from learn_bot.libs.temporal_column_names import get_temporal_field_str
from typing import Optional, Tuple
from math import pow


def compute_magnitude(df: pd.DataFrame, x_col: str, y_col: str, result_col: str, t: int):
    df[result_col] = \
        (df[get_temporal_field_str(x_col, t)].pow(2) + df[get_temporal_field_str(y_col, t)].pow(2)).pow(0.5)


def compute_per_axis_position_difference(df: pd.DataFrame, x_col: str, y_col: str, result_x_col: str, result_y_col: str,
                                start_t: int, end_t: int):
    df[result_x_col] = (df[get_temporal_field_str(x_col, end_t)] - df[get_temporal_field_str(x_col, start_t)]).abs()
    # modify x to deal with wrap around 180
    df[result_x_col] = df[result_x_col].where(df[result_x_col] < 180., 360. - df[result_x_col])
    df[result_y_col] = df[get_temporal_field_str(y_col, end_t)] - df[get_temporal_field_str(y_col, start_t)]


def compute_position_difference(df: pd.DataFrame, x_col: str, y_col: str, result_col: str,
                                start_t: int, end_t: int):
    x_distance = (df[get_temporal_field_str(x_col, end_t)] - df[get_temporal_field_str(x_col, start_t)]).abs()
    # modify x to deal with wrap around 180
    x_distance = x_distance.where(x_distance < 180., 360. - x_distance)
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

    selected_x_distance = (selected_row_df[x_end_col] - selected_row_df[x_start_col]).item()
    selected_y_distance = (selected_row_df[y_end_col] - selected_row_df[y_start_col]).item()
    selected_magnitude = pow(pow(selected_x_distance, 2) + pow(selected_y_distance, 2), 0.5)

    df[result_col] = np.rad2deg(np.arccos(
        ((x_distance * selected_x_distance) + (y_distance * selected_y_distance)) /
        (magnitude * selected_magnitude)))


default_speed_ticks = 5
@dataclass
class SimilarityConstraints:
    next_move_ticks: int
    same_alive: bool
    same_visibility: bool
    view_relative_to_enemy_radius: float
    mouse_speed_radius: float
    base_abs_view_angle_x_col: str
    base_abs_view_angle_y_col: str
    base_relative_view_angle_x_col: str
    base_relative_view_angle_y_col: str
    speed_direction_mouse_ticks: int = default_speed_ticks


@dataclass
class SimilarTrajectory:
    engagement_id: int
    tick_id: int

    def to_tuple(self) -> Tuple[int, int]:
        return self.engagement_id, self.tick_id


def compute_range_condition(derived_df: pd.DataFrame, selected_row_df: pd.DataFrame, x_col: str, y_col: str,
                            magnitude_col: str, radius_constraint: float):
    compute_magnitude(derived_df, x_col, y_col, magnitude_col, 0)
    compute_magnitude(selected_row_df, x_col, y_col, magnitude_col, 0)
    magnitude_radius_condition = (derived_df[magnitude_col] - selected_row_df[magnitude_col].item()).abs() <= \
                                 radius_constraint
    x_t_col = get_temporal_field_str(x_col, 0)
    y_t_col = get_temporal_field_str(x_col, 0)
    aabb_condition = \
        ((derived_df[x_t_col] - selected_row_df[x_t_col].item()).abs() <= radius_constraint) & \
        ((derived_df[y_t_col] - selected_row_df[y_t_col].item()).abs() <= radius_constraint)
    return magnitude_radius_condition & aabb_condition


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
        magnitude_col = "mouse distance to enemy magnitude"
        conditions = conditions & compute_range_condition(derived_df, selected_row_df,
                                                          constraints.base_relative_view_angle_x_col,
                                                          constraints.base_relative_view_angle_y_col,
                                                          magnitude_col, constraints.view_relative_to_enemy_radius)
    if constraints.mouse_speed_radius >= 0.:
        x_speed_col = "x mouse speed"
        y_speed_col = "y mouse speed"
        # need these to match naming approach of above
        x_t_speed_col = "x mouse speed (t)"
        y_t_speed_col = "y mouse speed (t)"
        magnitude_speed_col = "magnitude mouse speed"
        compute_per_axis_position_difference(derived_df,
                                             constraints.base_abs_view_angle_x_col,
                                             constraints.base_abs_view_angle_y_col,
                                             x_t_speed_col, y_t_speed_col,
                                             -1 * constraints.speed_direction_mouse_ticks, 0)
        compute_per_axis_position_difference(selected_row_df,
                                             constraints.base_abs_view_angle_x_col,
                                             constraints.base_abs_view_angle_y_col,
                                             x_t_speed_col, y_t_speed_col,
                                             -1 * constraints.speed_direction_mouse_ticks, 0)
        conditions = conditions & compute_range_condition(derived_df, selected_row_df,
                                                          x_speed_col, y_speed_col,
                                                          magnitude_speed_col, constraints.mouse_speed_radius)

    similar_df = derived_df[conditions]
    similar_engagements = similar_df.groupby([engagement_id_column]).agg({
        engagement_id_column: 'min',
        tick_id_column: 'min'
    })
    for _, row in similar_engagements.iterrows():
        result.append(SimilarTrajectory(row[engagement_id_column], row[tick_id_column]))

    return result



def remove_window():
    global child_window
    child_window.destroy()
    child_window = None


child_window: Optional[tk.Toplevel] = None
child_canvas: Optional[FigureCanvasTkAgg] = None
child_figure: Optional[Figure] = None
def plot_similar_trajectories_next_movement(parent_window: tk.Tk, not_selected_df: pd.DataFrame,
                                            similarity_constraints: SimilarityConstraints,
                                            similar_trajectories: List[SimilarTrajectory]):
    global child_window, child_canvas, child_figure
    if child_window is None:
        child_figure = Figure(figsize=(5.5, 5.5), dpi=100)

        child_window = tk.Toplevel(parent_window)
        child_window.protocol('WM_DELETE_WINDOW', remove_window)

        child_canvas = FigureCanvasTkAgg(child_figure, master=child_window)  # A tk.DrawingArea.
        child_canvas.draw()
        child_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # pack_toolbar=False will make it easier to use a layout manager later on.
        child_toolbar = NavigationToolbar2Tk(child_canvas, child_window, pack_toolbar=False)
        child_toolbar.update()
        child_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    child_figure.clear()
    ax = child_figure.gca()
    ax.invert_xaxis()

    similar_trajectories_tuples = [st.to_tuple() for st in similar_trajectories]
    trajectory_state_column = "trajectory states"
    similarity_points_df = not_selected_df.copy()
    similarity_points_df[trajectory_state_column] = list(zip(similarity_points_df[engagement_id_column],
                                                             similarity_points_df[tick_id_column]))
    similarity_points_df = \
        similarity_points_df[similarity_points_df[trajectory_state_column].isin(similar_trajectories_tuples)]

    x_pos_delta_col = "X Delta"
    y_pos_delta_col = "Y Delta"
    compute_per_axis_position_difference(similarity_points_df, similarity_constraints.base_abs_view_angle_x_col,
                                         similarity_constraints.base_abs_view_angle_y_col,
                                         x_pos_delta_col, y_pos_delta_col,
                                         0, similarity_constraints.next_move_ticks)

    heatmap, xedges, yedges = np.histogram2d(similarity_points_df[x_pos_delta_col],
                                             similarity_points_df[y_pos_delta_col], bins=50)

    # Histogram does not follow Cartesian convention (see Notes),
    # therefore transpose H for visualization purposes.
    heatmap = heatmap.T

    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, heatmap)
    child_figure.colorbar(im, ax=ax)
    ax.set_title(f"{len(similar_trajectories)} Similar Trajectories {similarity_constraints.next_move_ticks} Delta Pos")

    child_canvas.draw()

