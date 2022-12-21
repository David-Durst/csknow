from typing import Optional

import pandas as pd
import numpy as np
from matplotlib.legend_handler import HandlerLine2D

from learn_bot.engagement_aim.dataset import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from dataclasses import dataclass, field

from learn_bot.engagement_aim.vis.vis_similar_trajectories import compute_position_difference
from learn_bot.libs.temporal_column_names import get_temporal_field_str

# line colors
all_gray = (0.87, 0.87, 0.87, 1)
present_series_yellow = "#A89932FF"
prior_blue = "#00D5FAFF"
future_gray = "#727272FF"
present_red = (1., 0., 0., 1.)
fire_attack_hit_white = (1., 1., 1., 1.)
fire_attack_black = (0., 0., 0., 1.)
aabb_green = (0., 1., 0., 0.5)
aabb_blue = (0., 0., 0.61, 0.5)
aabb_black = (0., 0., 0., 1.0)
recoil_pink = (1., 0., 1., 1.)

# very unscientific head scale relative to rest of body
head_scale = 1. / 6.


class PerspectiveColumns:
    base_cur_view_angle_x_column: str
    base_cur_view_angle_x_column: str
    cur_view_angle_x_column: str
    cur_view_angle_y_column: str
    all_view_angle_x_columns: List[str]
    all_view_angle_y_columns: List[str]
    victim_min_view_angle_x_column: str
    victim_min_view_angle_y_column: str
    victim_max_view_angle_x_column: str
    victim_max_view_angle_y_column: str
    victim_cur_head_view_angle_x_column: str
    victim_cur_head_view_angle_y_column: str
    recoil_x_column: str
    recoil_y_column: str

    def __init__(self, x_col, victim_min_view_angle_x_col, recoil_x_col):
        x_col_offset = base_vis_float_columns.index(x_col)
        y_col_offset = x_col_offset + 1
        victim_min_view_angle_x_offset = base_vis_float_columns.index(victim_min_view_angle_x_col)
        recoil_offset = base_vis_float_columns.index(recoil_x_col)

        self.base_cur_view_angle_x_column = base_vis_float_columns[x_col_offset]
        self.base_cur_view_angle_y_column = base_vis_float_columns[y_col_offset]
        self.cur_view_angle_x_column = temporal_vis_float_column_names.present_columns[x_col_offset]
        self.cur_view_angle_y_column = temporal_vis_float_column_names.present_columns[y_col_offset]
        self.all_view_angle_x_columns = \
            temporal_vis_float_column_names.get_matching_cols(base_vis_float_columns[x_col_offset], include_future=False)
        self.all_view_angle_y_columns = \
            temporal_vis_float_column_names.get_matching_cols(base_vis_float_columns[y_col_offset], include_future=False)
        self.victim_min_view_angle_x_column = \
            temporal_vis_float_column_names.present_columns[victim_min_view_angle_x_offset]
        self.victim_min_view_angle_y_column = \
            temporal_vis_float_column_names.present_columns[victim_min_view_angle_x_offset + 1]
        self.victim_max_view_angle_x_column = \
            temporal_vis_float_column_names.present_columns[victim_min_view_angle_x_offset + 2]
        self.victim_max_view_angle_y_column = \
            temporal_vis_float_column_names.present_columns[victim_min_view_angle_x_offset + 3]
        self.victim_cur_head_view_angle_x_column = \
            temporal_vis_float_column_names.present_columns[victim_min_view_angle_x_offset + 4]
        self.victim_cur_head_view_angle_y_column = \
            temporal_vis_float_column_names.present_columns[victim_min_view_angle_x_offset + 5]
        self.recoil_x_column = \
            temporal_vis_float_column_names.present_columns[recoil_offset]
        self.recoil_y_column = \
            temporal_vis_float_column_names.present_columns[recoil_offset + 1]


class DataFrameTemporalSlices:
    prior_df: pd.DataFrame
    prior_x_np: np.ndarray
    prior_y_np: np.ndarray

    present_df: pd.DataFrame
    present_x_np: np.ndarray
    present_y_np: np.ndarray

    # shows series used in prediction, not just single point
    present_series_df: Optional[pd.DataFrame]
    present_series_x_np: Optional[np.ndarray]
    present_series_y_np: Optional[np.ndarray]

    future_df: pd.DataFrame
    future_x_np: np.ndarray
    future_y_np: np.ndarray

    all_x_np: np.ndarray
    all_y_np: np.ndarray

    fire_df: pd.DataFrame
    fire_x_np: np.ndarray
    fire_y_np: np.ndarray

    hold_attack_df: pd.DataFrame
    hold_attack_x_np: np.ndarray
    hold_attack_y_np: np.ndarray

    hit_victim_df: pd.DataFrame
    hit_victim_x_np: np.ndarray
    hit_victim_y_np: np.ndarray

    def __init__(self, data_df: pd.DataFrame, tick_id: int, columns: PerspectiveColumns,
                 x_col: str, y_col: str, include_present_series: bool):
        # pos data
        self.prior_df = data_df[data_df['tick id'] < tick_id]
        self.prior_x_np = self.prior_df.loc[:, x_col].to_numpy()
        self.prior_y_np = self.prior_df.loc[:, y_col].to_numpy()

        self.present_df = data_df[data_df['tick id'] == tick_id]
        self.present_x_np = self.present_df.loc[:, x_col].to_numpy()
        self.present_y_np = self.present_df.loc[:, y_col].to_numpy()

        # shows series used in prediction, not just single point
        if include_present_series:
            self.present_series_df = data_df[data_df['tick id'] == tick_id].iloc[0, :]
            self.present_series_x_np = self.present_series_df.loc[columns.all_view_angle_x_columns].to_numpy()
            self.present_series_y_np = self.present_series_df.loc[columns.all_view_angle_y_columns].to_numpy()
        else:
            self.present_series_df = None
            self.present_series_x_np = None
            self.present_series_y_np = None

        self.future_df = data_df[data_df['tick id'] > tick_id]
        self.future_x_np = self.future_df.loc[:, x_col].to_numpy()
        self.future_y_np = self.future_df.loc[:, y_col].to_numpy()

        self.all_x_np = data_df.loc[:, x_col].to_numpy()
        self.all_y_np = data_df.loc[:, y_col].to_numpy()

        self.fire_df = data_df[data_df['ticks until next fire (t)'] == 0.]
        self.fire_x_np = self.fire_df.loc[:, x_col].to_numpy()
        self.fire_y_np = self.fire_df.loc[:, y_col].to_numpy()

        self.hold_attack_df = data_df[data_df['ticks until next holding attack (t)'] == 0.]
        self.hold_attack_x_np = self.hold_attack_df.loc[:, x_col].to_numpy()
        self.hold_attack_y_np = self.hold_attack_df.loc[:, y_col].to_numpy()

        self.hit_victim_df = data_df[data_df['hit victim (t)'] == 1.]
        self.hit_victim_x_np = self.hit_victim_df.loc[:, x_col].to_numpy()
        self.hit_victim_y_np = self.hit_victim_df.loc[:, y_col].to_numpy()

no_legend_str = "_nolegend_"

class TemporalLines:
    # pos ax lines
    all_line: Line2D
    prior_line: Line2D
    present_line: Line2D
    present_series_line: Optional[Line2D]
    future_line: Line2D
    fire_line: Optional[Line2D]
    hold_attack_line: Optional[Line2D]
    hit_victim_line: Optional[Line2D]

    def __init__(self, ax: plt.Axes, df_temporal_slices: DataFrameTemporalSlices, include_present_series: bool,
                 use_legend: bool):
        self.all_line, = ax.plot(df_temporal_slices.all_x_np, df_temporal_slices.all_y_np,
                                 color=all_gray, label=no_legend_str)
        if include_present_series:
            self.present_series_line, = ax.plot(df_temporal_slices.present_series_x_np,
                                                df_temporal_slices.present_series_y_np,
                                                linestyle="None", label="Model Feature" if use_legend else no_legend_str,
                                                marker='o', mfc="None", mec=present_series_yellow,
                                                markersize=10)
        else:
            self.present_series_line = None
        self.prior_line, = ax.plot(df_temporal_slices.prior_x_np, df_temporal_slices.prior_y_np,
                                   linestyle="None", label="Past" if use_legend else no_legend_str,
                                   marker='o', mfc=prior_blue, mec=prior_blue)
        self.future_line, = ax.plot(df_temporal_slices.future_x_np, df_temporal_slices.future_y_np,
                                    linestyle="None", label="Future" if use_legend else no_legend_str,
                                    marker='o', mfc=future_gray, mec=future_gray)
        self.present_line, = ax.plot(df_temporal_slices.present_x_np, df_temporal_slices.present_y_np,
                                     linestyle="None", label="Present" if use_legend else no_legend_str,
                                     marker='o', mfc=present_red, mec=present_red)
        self.fire_line, = ax.plot(df_temporal_slices.fire_x_np, df_temporal_slices.fire_y_np,
                                  linestyle="None", label="Fire" if use_legend else no_legend_str,
                                  marker='|', mfc=fire_attack_hit_white, mec=fire_attack_hit_white)
        self.hold_attack_line, = ax.plot(df_temporal_slices.hold_attack_x_np, df_temporal_slices.hold_attack_y_np,
                                         linestyle="None", label="Attack" if use_legend else no_legend_str,
                                         marker='_', mfc=fire_attack_hit_white, mec=fire_attack_hit_white)
        self.hit_victim_line, = ax.plot(df_temporal_slices.hit_victim_x_np, df_temporal_slices.hit_victim_y_np,
                                        linestyle="None", label="Hit" if use_legend else no_legend_str,
                                        marker='x', mfc=fire_attack_hit_white, mec=fire_attack_hit_white)

    def update(self, df_temporal_slices: DataFrameTemporalSlices):
        self.all_line.set_data(df_temporal_slices.all_x_np,
                               df_temporal_slices.all_y_np)
        if self.present_series_line is not None:
            self.present_series_line.set_data(df_temporal_slices.present_series_x_np,
                                              df_temporal_slices.present_series_y_np)
        self.prior_line.set_data(df_temporal_slices.prior_x_np,
                                 df_temporal_slices.prior_y_np)
        self.future_line.set_data(df_temporal_slices.future_x_np,
                                  df_temporal_slices.future_y_np)
        self.present_line.set_data(df_temporal_slices.present_x_np,
                                   df_temporal_slices.present_y_np)
        self.fire_line.set_data(df_temporal_slices.fire_x_np,
                                df_temporal_slices.fire_y_np)
        self.hold_attack_line.set_data(df_temporal_slices.hold_attack_x_np,
                                       df_temporal_slices.hold_attack_y_np)
        self.hit_victim_line.set_data(df_temporal_slices.hit_victim_x_np,
                                      df_temporal_slices.hit_victim_y_np)

    def remove(self):
        self.all_line.remove()
        if self.present_series_line is not None:
            self.present_series_line.remove()
        self.prior_line.remove()
        self.future_line.remove()
        self.present_line.remove()
        self.fire_line.remove()
        self.hold_attack_line.remove()
        self.hit_victim_line.remove()

# https://stackoverflow.com/questions/11690597/there-is-a-class-matplotlib-axes-axessubplot-but-the-module-matplotlib-axes-has
# matplotlib.axes._subplots.AxesSubplot doesn't exist statically
# https://stackoverflow.com/questions/11690597/there-is-a-class-matplotlib-axes-axessubplot-but-the-module-matplotlib-axes-has
# so use this instead
@dataclass
class AxObjs:
    fig: plt.Figure
    pos_ax: plt.Axes
    speed_ax: plt.Axes
    first_tick_columns: PerspectiveColumns
    cur_head_columns: PerspectiveColumns
    pos_temporal_lines: Optional[TemporalLines] = None
    speed_temporal_lines: Optional[TemporalLines] = None
    pos_recoil_line: Optional[Line2D] = None
    pos_victim_head_circle: Optional[Circle] = None
    pos_victim_aabb: Optional[Rectangle] = None

    def update_aim_plot(self, selected_df: pd.DataFrame, tick_id: int, canvas: FigureCanvasTkAgg, use_first_tick: bool):
        columns = self.first_tick_columns if use_first_tick else self.cur_head_columns
        pos_df_temporal_slices = DataFrameTemporalSlices(selected_df, tick_id, columns,
                                                         columns.cur_view_angle_x_column,
                                                         columns.cur_view_angle_y_column,
                                                         True)

        pos_recoil_x_np = pos_df_temporal_slices.present_df.loc[:, columns.recoil_x_column].to_numpy() + \
                          pos_df_temporal_slices.present_x_np
        pos_recoil_y_np = pos_df_temporal_slices.present_df.loc[:, columns.recoil_y_column].to_numpy() + \
                          pos_df_temporal_slices.present_y_np

        aabb_min = (
            pos_df_temporal_slices.present_df.loc[:, columns.victim_min_view_angle_x_column].item(),
            pos_df_temporal_slices.present_df.loc[:, columns.victim_min_view_angle_y_column].item()
        )
        aabb_max = (
            pos_df_temporal_slices.present_df.loc[:, columns.victim_max_view_angle_x_column].item(),
            pos_df_temporal_slices.present_df.loc[:, columns.victim_max_view_angle_y_column].item()
        )
        aabb_size = (
            aabb_max[0] - aabb_min[0],
            aabb_max[1] - aabb_min[1]
        )
        head_center = (
            pos_df_temporal_slices.present_df.loc[:, columns.victim_cur_head_view_angle_x_column].item(),
            pos_df_temporal_slices.present_df.loc[:, columns.victim_cur_head_view_angle_y_column].item()
        )
        head_radius = (aabb_max[0] - aabb_min[0]) / 2. * head_scale
        victim_visible = pos_df_temporal_slices.present_df.loc[:, cur_victim_visible_column].item()
        victim_alive = pos_df_temporal_slices.present_df.loc[:, cur_victim_alive_column].item()

        # speed data
        speed_cols = []
        min_game_time = min(selected_df['game time'])
        speed_df = selected_df.copy()
        speed_df['delta game time'] = (selected_df['game time'] - min_game_time) / 1000.
        for i in range(-4, 1):
            speed_cols.append(get_temporal_field_str("speed at", i))
            compute_position_difference(speed_df, columns.base_cur_view_angle_x_column, columns.base_cur_view_angle_y_column,
                                        speed_cols[-1], i - 1, i)
        median_speed_col = "median speed"
        speed_df[median_speed_col] = speed_df[speed_cols].mean(axis=1)

        speed_df_temporal_slices = DataFrameTemporalSlices(speed_df, tick_id, columns,
                                                           "delta game time", median_speed_col,
                                                           False)

        if self.pos_temporal_lines is None:
            # ax1.plot(x, y,color='#FF0000', linewidth=2.2, label='Example line',
            #           marker='o', mfc='black', mec='black', ms=10)
            self.pos_recoil_line, = self.pos_ax.plot(pos_recoil_x_np, pos_recoil_y_np, linestyle="None", label="Recoil",
                                                     marker='o', mfc=recoil_pink, mec=recoil_pink)
            self.pos_temporal_lines = TemporalLines(self.pos_ax, pos_df_temporal_slices, True, True)
            self.pos_victim_aabb = Rectangle(aabb_min, aabb_size[0], aabb_size[1],
                                             linewidth=2, edgecolor=aabb_green, facecolor='none')
            self.pos_ax.add_patch(self.pos_victim_aabb)
            self.pos_victim_head_circle = Circle(head_center, head_radius, color=aabb_green)
            self.pos_ax.add_patch(self.pos_victim_head_circle)
            self.speed_temporal_lines = TemporalLines(self.speed_ax, speed_df_temporal_slices, False, False)
        else:
            self.pos_recoil_line.set_data(pos_recoil_x_np,
                                          pos_recoil_y_np)
            self.pos_temporal_lines.update(pos_df_temporal_slices)
            self.pos_victim_aabb.set_xy(aabb_min)
            self.pos_victim_aabb.set_width(aabb_size[0])
            self.pos_victim_aabb.set_height(aabb_size[1])
            self.pos_victim_head_circle.set_center(head_center)
            self.pos_victim_head_circle.set_radius(head_radius)
            self.speed_temporal_lines.update(speed_df_temporal_slices)

        if not victim_alive:
            self.pos_victim_aabb.set_edgecolor(aabb_black)
            self.pos_victim_head_circle.set_color(aabb_black)
        elif victim_visible:
            self.pos_victim_aabb.set_edgecolor(aabb_green)
            self.pos_victim_head_circle.set_color(aabb_green)
        else:
            self.pos_victim_aabb.set_edgecolor(aabb_blue)
            self.pos_victim_head_circle.set_color(aabb_blue)

        # recompute the ax.dataLim
        self.pos_ax.relim()
        # update ax.viewLim using the new dataLim
        self.pos_ax.autoscale()
        legend = self.pos_ax.legend()
        for handle in legend.legendHandles:
            if handle.get_label() in ["Fire", "Attack", "Hit"]:
                handle.set_mfc(fire_attack_black)
                handle.set_mec(fire_attack_black)

        x_max, x_min = self.pos_ax.get_xlim()
        y_min, y_max = self.pos_ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range < y_range:
            range_diff = y_range - x_range
            x_min -= range_diff / 2.
            x_max += range_diff / 2.
        elif y_range < x_range:
            range_diff = x_range - y_range
            y_min -= range_diff / 2.
            y_max += range_diff / 2.
        # inverted xaxis so need to flip
        self.pos_ax.set_xlim(x_max, x_min)
        self.pos_ax.set_ylim(y_min, y_max)

        # rescale speed axis, just make 0 is start
        self.speed_ax.relim()
        self.speed_ax.autoscale()
        self.speed_ax.set_xlim(left=0)
        self.speed_ax.set_ylim(bottom=0)

        # required to update canvas and attached toolbar!
        canvas.draw()
