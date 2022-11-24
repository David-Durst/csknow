from typing import Optional

import pandas as pd
import numpy as np
from matplotlib.legend_handler import HandlerLine2D

from learn_bot.engagement_aim.dataset import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from dataclasses import dataclass

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

    def __init__(self, x_col_offset, victim_min_view_angle_x_offset, recoil_offset):
        y_col_offset = x_col_offset + 1
        self.base_cur_view_angle_x_column = base_float_columns[x_col_offset]
        self.base_cur_view_angle_y_column = base_float_columns[y_col_offset]
        self.cur_view_angle_x_column = temporal_io_float_column_names.vis_columns[x_col_offset]
        self.cur_view_angle_y_column = temporal_io_float_column_names.vis_columns[y_col_offset]
        self.all_view_angle_x_columns = \
            temporal_io_float_column_names.get_matching_cols(base_float_columns[x_col_offset])
        self.all_view_angle_y_columns = \
            temporal_io_float_column_names.get_matching_cols(base_float_columns[y_col_offset])
        self.victim_min_view_angle_x_column = \
            temporal_io_float_column_names.vis_columns[victim_min_view_angle_x_offset]
        self.victim_min_view_angle_y_column = \
            temporal_io_float_column_names.vis_columns[victim_min_view_angle_x_offset + 1]
        self.victim_max_view_angle_x_column = \
            temporal_io_float_column_names.vis_columns[victim_min_view_angle_x_offset + 2]
        self.victim_max_view_angle_y_column = \
            temporal_io_float_column_names.vis_columns[victim_min_view_angle_x_offset + 3]
        self.victim_cur_head_view_angle_x_column = \
            temporal_io_float_column_names.vis_columns[victim_min_view_angle_x_offset + 4]
        self.victim_cur_head_view_angle_y_column = \
            temporal_io_float_column_names.vis_columns[victim_min_view_angle_x_offset + 5]
        self.recoil_x_column = \
            temporal_io_float_column_names.vis_columns[recoil_offset]
        self.recoil_y_column = \
            temporal_io_float_column_names.vis_columns[recoil_offset + 1]


class DataFrameTemporalSlices:
    pos_prior_df: pd.DataFrame
    pos_prior_x_np: np.ndarray
    pos_prior_y_np: np.ndarray

    pos_present_df: pd.DataFrame
    pos_present_x_np: np.ndarray
    pos_present_y_np: np.ndarray

    # shows series used in prediction, not just single point
    pos_present_series_df: Optional[pd.DataFrame]
    pos_present_series_x_np: Optional[np.ndarray]
    pos_present_series_y_np: Optional[np.ndarray]

    pos_future_df: pd.DataFrame
    pos_future_x_np: np.ndarray
    pos_future_y_np: np.ndarray

    pos_all_x_np: np.ndarray
    pos_all_y_np: np.ndarray

    def __init__(self, data_df: pd.DataFrame, tick_id: int, columns: PerspectiveColumns,
                 x_col: str, y_col: str, include_present_series: bool):
        # pos data
        self.pos_prior_df = data_df[data_df['tick id'] < tick_id]
        self.pos_prior_x_np = self.pos_prior_df.loc[:, x_col].to_numpy()
        self.pos_prior_y_np = self.pos_prior_df.loc[:, y_col].to_numpy()

        self.pos_present_df = data_df[data_df['tick id'] == tick_id]
        self.pos_present_x_np = self.pos_present_df.loc[:, x_col].to_numpy()
        self.pos_present_y_np = self.pos_present_df.loc[:, y_col].to_numpy()

        # shows series used in prediction, not just single point
        if include_present_series:
            self.pos_present_series_df = data_df[data_df['tick id'] == tick_id].iloc[0, :]
            self.pos_present_series_x_np = self.pos_present_series_df.loc[columns.all_view_angle_x_columns].to_numpy()
            self.pos_present_series_y_np = self.pos_present_series_df.loc[columns.all_view_angle_y_columns].to_numpy()
        else:
            self.pos_present_series_df = None
            self.pos_present_series_x_np = None
            self.pos_present_series_y_np = None

        self.pos_future_df = data_df[data_df['tick id'] > tick_id]
        self.pos_future_x_np = self.pos_future_df.loc[:, x_col].to_numpy()
        self.pos_future_y_np = self.pos_future_df.loc[:, y_col].to_numpy()

        self.pos_all_x_np = data_df.loc[:, x_col].to_numpy()
        self.pos_all_y_np = data_df.loc[:, y_col].to_numpy()


class TemporalLines:
    # pos ax lines
    pos_all_line: Line2D
    pos_prior_line: Line2D
    pos_present_line: Line2D
    pos_present_series_line: Optional[Line2D]
    pos_future_line: Line2D

    def __init__(self, ax: plt.Axes, df_temporal_slices: DataFrameTemporalSlices, include_present_series: bool):
        self.pos_all_line, = ax.plot(df_temporal_slices.pos_all_x_np, df_temporal_slices.pos_all_y_np,
                                     color=all_gray, label="_nolegend_")
        if include_present_series:
            self.pos_present_series_line, = ax.plot(df_temporal_slices.pos_present_series_x_np,
                                                    df_temporal_slices.pos_present_series_y_np,
                                                    linestyle="None", label="Model Feature",
                                                    marker='o', mfc="None", mec=present_series_yellow,
                                                    markersize=10)
        else:
            self.pos_present_series_line = None
        self.pos_prior_line, = ax.plot(df_temporal_slices.pos_prior_x_np, df_temporal_slices.pos_prior_y_np,
                                       linestyle="None", label="Past",
                                       marker='o', mfc=prior_blue, mec=prior_blue)
        self.pos_future_line, = ax.plot(df_temporal_slices.pos_future_x_np, df_temporal_slices.pos_future_y_np,
                                        linestyle="None", label="Future",
                                        marker='o', mfc=future_gray, mec=future_gray)
        self.pos_present_line, = ax.plot(df_temporal_slices.pos_present_x_np, df_temporal_slices.pos_present_y_np,
                                         linestyle="None", label="Present",
                                         marker='o', mfc=present_red, mec=present_red)

    def update(self, df_temporal_slices: DataFrameTemporalSlices):
        self.pos_all_line.set_data(df_temporal_slices.pos_all_x_np,
                                   df_temporal_slices.pos_all_y_np)
        if self.pos_present_series_line is not None:
            self.pos_present_series_line.set_data(df_temporal_slices.pos_present_series_x_np,
                                                  df_temporal_slices.pos_present_series_y_np)
        self.pos_prior_line.set_data(df_temporal_slices.pos_prior_x_np,
                                     df_temporal_slices.pos_prior_y_np)
        self.pos_future_line.set_data(df_temporal_slices.pos_future_x_np,
                                      df_temporal_slices.pos_future_y_np)
        self.pos_present_line.set_data(df_temporal_slices.pos_present_x_np,
                                       df_temporal_slices.pos_present_y_np)


# https://stackoverflow.com/questions/11690597/there-is-a-class-matplotlib-axes-axessubplot-but-the-module-matplotlib-axes-has
# matplotlib.axes._subplots.AxesSubplot doesn't exist statically
# https://stackoverflow.com/questions/11690597/there-is-a-class-matplotlib-axes-axessubplot-but-the-module-matplotlib-axes-has
# so use this instead
@dataclass
class AxObjs:
    fig: plt.Figure
    pos_ax: plt.Axes
    speed_ax: plt.Axes
    first_hit_columns: PerspectiveColumns
    cur_head_columns: PerspectiveColumns
    pos_temporal_lines: Optional[TemporalLines] = None
    speed_temporal_lines: Optional[TemporalLines] = None
    pos_fire_line: Optional[Line2D] = None
    pos_hold_attack_line: Optional[Line2D] = None
    pos_hit_victim_line: Optional[Line2D] = None
    pos_recoil_line: Optional[Line2D] = None
    pos_victim_head_circle: Optional[Circle] = None
    pos_victim_aabb: Optional[Rectangle] = None

    def update_aim_plot(self, data_df: pd.DataFrame, tick_id: int, canvas: FigureCanvasTkAgg, use_first_hit: bool):
        columns = self.first_hit_columns if use_first_hit else self.cur_head_columns
        pos_df_temporal_slices = DataFrameTemporalSlices(data_df, tick_id, columns,
                                                         columns.cur_view_angle_x_column,
                                                         columns.cur_view_angle_y_column,
                                                         True)

        pos_fire_df = data_df[data_df['ticks until next fire (t)'] == 0.]
        pos_fire_x_np = pos_fire_df.loc[:, columns.cur_view_angle_x_column].to_numpy()
        pos_fire_y_np = pos_fire_df.loc[:, columns.cur_view_angle_y_column].to_numpy()

        pos_hold_attack_df = data_df[data_df['ticks until next holding attack (t)'] == 0.]
        pos_hold_attack_x_np = pos_hold_attack_df.loc[:, columns.cur_view_angle_x_column].to_numpy()
        pos_hold_attack_y_np = pos_hold_attack_df.loc[:, columns.cur_view_angle_y_column].to_numpy()

        pos_hit_victim_df = data_df[data_df['hit victim (t)'] == 1.]
        pos_hit_victim_x_np = pos_hit_victim_df.loc[:, columns.cur_view_angle_x_column].to_numpy()
        pos_hit_victim_y_np = pos_hit_victim_df.loc[:, columns.cur_view_angle_y_column].to_numpy()

        pos_recoil_x_np = pos_df_temporal_slices.pos_present_df.loc[:, columns.recoil_x_column].to_numpy() + \
                          pos_df_temporal_slices.pos_present_x_np
        pos_recoil_y_np = pos_df_temporal_slices.pos_present_df.loc[:, columns.recoil_y_column].to_numpy() + \
                          pos_df_temporal_slices.pos_present_y_np

        aabb_min = (
            pos_df_temporal_slices.pos_present_df.loc[:, columns.victim_min_view_angle_x_column].item(),
            pos_df_temporal_slices.pos_present_df.loc[:, columns.victim_min_view_angle_y_column].item()
        )
        aabb_max = (
            pos_df_temporal_slices.pos_present_df.loc[:, columns.victim_max_view_angle_x_column].item(),
            pos_df_temporal_slices.pos_present_df.loc[:, columns.victim_max_view_angle_y_column].item()
        )
        aabb_size = (
            aabb_max[0] - aabb_min[0],
            aabb_max[1] - aabb_min[1]
        )
        head_center = (
            pos_df_temporal_slices.pos_present_df.loc[:, columns.victim_cur_head_view_angle_x_column].item(),
            pos_df_temporal_slices.pos_present_df.loc[:, columns.victim_cur_head_view_angle_y_column].item()
        )
        head_radius = (aabb_max[0] - aabb_min[0]) / 2. * head_scale
        victim_visible = pos_df_temporal_slices.pos_present_df.loc[:, 'victim visible (t)'].item()
        victim_alive = pos_df_temporal_slices.pos_present_df.loc[:, 'victim alive (t)'].item()

        # speed data
        speed_cols = []
        speed_df = pd.DataFrame()
        speed_df['tick id'] = data_df['tick id']
        for i in range(-1, 1):
            x_speed_col = get_temporal_field_str("x speed at", i)
            y_speed_col = get_temporal_field_str("y speed at", i)
            speed_cols.append(get_temporal_field_str("speed at", i))
            speed_df[x_speed_col] = \
                data_df[get_temporal_field_str(columns.base_cur_view_angle_x_column, i)] - \
                data_df[get_temporal_field_str(columns.base_cur_view_angle_x_column, i-1)]
            speed_df[y_speed_col] = \
                data_df[get_temporal_field_str(columns.base_cur_view_angle_y_column, i)] - \
                data_df[get_temporal_field_str(columns.base_cur_view_angle_y_column, i - 1)]
            speed_df[speed_cols[-1]] = \
                (speed_df[x_speed_col].pow(2) + speed_df[y_speed_col].pow(2)).pow(0.5)
        median_speed_col = "median speed"
        speed_df[median_speed_col] = speed_df[speed_cols].median(axis=1)
        speed_df_temporal_slices = DataFrameTemporalSlices(speed_df, tick_id, columns,
                                                           "tick id", median_speed_col,
                                                           False)

        if self.pos_temporal_lines is None:
            # ax1.plot(x, y,color='#FF0000', linewidth=2.2, label='Example line',
            #           marker='o', mfc='black', mec='black', ms=10)
            self.pos_recoil_line, = self.pos_ax.plot(pos_recoil_x_np, pos_recoil_y_np, linestyle="None", label="Recoil",
                                                     marker='o', mfc=recoil_pink, mec=recoil_pink)
            self.pos_temporal_lines = TemporalLines(self.pos_ax, pos_df_temporal_slices, True)
            self.pos_fire_line, = self.pos_ax.plot(pos_fire_x_np, pos_fire_y_np, linestyle="None", label="Fire",
                                                   marker='|', mfc=fire_attack_hit_white, mec=fire_attack_hit_white)
            self.pos_hold_attack_line, = self.pos_ax.plot(pos_hold_attack_x_np, pos_hold_attack_y_np,
                                                          linestyle="None", label="Attack",
                                                          marker='_', mfc=fire_attack_hit_white,
                                                          mec=fire_attack_hit_white)
            self.pos_hit_victim_line, = self.pos_ax.plot(pos_hit_victim_x_np, pos_hit_victim_y_np,
                                                         linestyle="None", label="Hit",
                                                         marker='x', mfc=fire_attack_hit_white,
                                                         mec=fire_attack_hit_white)
            self.pos_victim_aabb = Rectangle(aabb_min, aabb_size[0], aabb_size[1],
                                             linewidth=2, edgecolor=aabb_green, facecolor='none')
            self.pos_ax.add_patch(self.pos_victim_aabb)
            self.pos_victim_head_circle = Circle(head_center, head_radius, color=aabb_green)
            self.pos_ax.add_patch(self.pos_victim_head_circle)
            self.speed_temporal_lines = TemporalLines(self.speed_ax, speed_df_temporal_slices, False)
        else:
            self.pos_recoil_line.set_data(pos_recoil_x_np,
                                          pos_recoil_y_np)
            self.pos_temporal_lines.update(pos_df_temporal_slices)
            self.pos_fire_line.set_data(pos_fire_x_np, pos_fire_y_np)
            self.pos_hold_attack_line.set_data(pos_hold_attack_x_np, pos_hold_attack_y_np)
            self.pos_hit_victim_line.set_data(pos_hit_victim_x_np, pos_hit_victim_y_np)
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

        # required to update canvas and attached toolbar!
        canvas.draw()
