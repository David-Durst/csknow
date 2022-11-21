from typing import Optional

import pandas as pd
from pathlib import Path

from learn_bot.libs.df_grouping import make_index_column
from learn_bot.engagement_aim.dataset import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from dataclasses import dataclass

# line colors
all_gray = (0.87, 0.87, 0.87, 1)
present_series_yellow = "#A89932FF"
prior_blue = "#00D5FAFF"
future_gray = "#727272FF"
present_red = (1., 0., 0., 0.5)

class PerspectiveColumns:
    cur_x_column: str
    cur_y_column: str
    all_x_columns: List[str]
    all_y_columns: List[str]

    def __init__(self, x_col_offset, y_col_offset):
        self.cur_x_column = temporal_io_float_column_names.vis_columns[x_col_offset]
        self.cur_y_column = temporal_io_float_column_names.vis_columns[y_col_offset]
        self.all_x_columns = \
            temporal_io_float_column_names.get_matching_cols(base_float_columns[x_col_offset])
        self.all_y_columns = \
            temporal_io_float_column_names.get_matching_cols(base_float_columns[y_col_offset])


# https://stackoverflow.com/questions/11690597/there-is-a-class-matplotlib-axes-axessubplot-but-the-module-matplotlib-axes-has
# matplotlib.axes._subplots.AxesSubplot doesn't exist statically
# https://stackoverflow.com/questions/11690597/there-is-a-class-matplotlib-axes-axessubplot-but-the-module-matplotlib-axes-has
# so use this instead
@dataclass
class AxObjs:
    ax: plt.Axes
    first_hit_columns: PerspectiveColumns
    cur_head_columns: PerspectiveColumns
    all_line: Optional[Line2D] = None
    prior_line: Optional[Line2D] = None
    present_line: Optional[Line2D] = None
    present_series_line: Optional[Line2D] = None
    future_line: Optional[Line2D] = None

    def update_aim_plot(self, data_df: pd.DataFrame, tick_id: int, canvas: FigureCanvasTkAgg, use_first_hit: bool):
        columns = self.first_hit_columns if use_first_hit else self.cur_head_columns
        prior_df = data_df[data_df['tick id'] < tick_id]
        prior_x_np = prior_df.loc[:, columns.cur_x_column].to_numpy()
        prior_y_np = prior_df.loc[:, columns.cur_y_column].to_numpy()

        present_df = data_df[data_df['tick id'] == tick_id]
        present_x_np = present_df.loc[:, columns.cur_x_column].to_numpy()
        present_y_np = present_df.loc[:, columns.cur_y_column].to_numpy()

        # shows series used in prediction, not sust single point
        present_series_df = data_df[data_df['tick id'] == tick_id].iloc[0, :]
        present_series_x_np = present_series_df.loc[columns.all_x_columns].to_numpy()
        present_series_y_np = present_series_df.loc[columns.all_y_columns].to_numpy()

        future_df = data_df[data_df['tick id'] > tick_id]
        future_x_np = future_df.loc[:, columns.cur_x_column].to_numpy()
        future_y_np = future_df.loc[:, columns.cur_y_column].to_numpy()

        all_x_np = data_df.loc[:, columns.cur_x_column].to_numpy()
        all_y_np = data_df.loc[:, columns.cur_y_column].to_numpy()

        if self.prior_line is None:
            # ax1.plot(x, y,color='#FF0000', linewidth=2.2, label='Example line',
            #           marker='o', mfc='black', mec='black', ms=10)
            self.all_line, = self.ax.plot(all_x_np, all_y_np, color=all_gray, label="_nolegend_")
            self.present_series_line, = self.ax.plot(present_series_x_np, present_series_y_np,
                                                     linestyle="None", label="Model Feature",
                                                     marker='o', mfc="None", mec=present_series_yellow,
                                                     markersize=10)
            self.prior_line, = self.ax.plot(prior_x_np, prior_y_np, linestyle="None", label="Past",
                                            marker='o', mfc=prior_blue, mec=prior_blue)
            self.future_line, = self.ax.plot(future_x_np, future_y_np, linestyle="None", label="Future",
                                             marker='o', mfc=future_gray, mec=future_gray)
            self.present_line, = self.ax.plot(present_x_np, present_y_np, linestyle="None", label="Present",
                                              marker='o', mfc=present_red, mec=present_red)
        else:
            self.all_line.set_data(all_x_np, all_y_np)
            self.present_series_line.set_data(present_series_x_np, present_series_y_np)
            self.prior_line.set_data(prior_x_np, prior_y_np)
            self.future_line.set_data(future_x_np, future_y_np)
            self.present_line.set_data(present_x_np, present_y_np)

        # recompute the ax.dataLim
        self.ax.relim()
        # update ax.viewLim using the new dataLim
        self.ax.autoscale()
        self.ax.legend()
        x_max, x_min = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
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
        self.ax.set_xlim(x_max, x_min)
        self.ax.set_ylim(y_min, y_max)

        # required to update canvas and attached toolbar!
        canvas.draw()



def vis(all_data_df: pd.DataFrame, pred_df: pd.DataFrame = None):
    all_data_df = all_data_df.sort_values(['engagement id', 'tick id'])
    make_index_column(all_data_df)

    if pred_df is not None:
        pred_df = pred_df.sort_values(['engagement id', 'tick id'])
        make_index_column(pred_df)

    #This creates the main window of an application
    window = tk.Tk()
    window.title("Aim Images")
    if pred_df is None:
        window.geometry("650x950")
    else:
        window.geometry("1100x950")
    window.configure(background='grey')

    # columns for reading d
    first_hit_columns = PerspectiveColumns(4, 5)
    cur_hit_columns = PerspectiveColumns(6, 7)

    # create axes and their objects
    first_hit_title_suffix = " Relative To First Hit Enemy Head"
    cur_pos_title_suffix = " Relative To Cur Enemy Head"
    def setAxSettings(ax: plt.Axes, title: str):
        ax.base_title = title
        ax.set_title(title + first_hit_title_suffix)
        ax.set_xlabel("Yaw Delta (deg)")
        ax.set_ylabel("Pitch Delta (deg)")
        ax.set_aspect('equal', adjustable='box')
        ax.invert_xaxis()

    if pred_df is None:
        fig = Figure(figsize=(5.5, 5.5), dpi=100)
        input_ax = fig.add_subplot()
    else:
        fig = Figure(figsize=(11., 5.5), dpi=100)
        input_ax, pred_ax = fig.subplots(nrows=1, ncols=2)
        setAxSettings(pred_ax, "Pred Aim Data")
        pred_ax_objs = AxObjs(pred_ax, first_hit_columns, cur_hit_columns)
    setAxSettings(input_ax, "Input Aim Data")
    input_ax_objs = AxObjs(input_ax, first_hit_columns, cur_hit_columns)

    canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    engagements = all_data_df.loc[:, 'engagement id'].unique().tolist()
    indices = []
    ticks = []
    demo_ticks = []
    game_ticks = []
    cur_engagement: int = -1
    cur_tick: int = -1
    selected_df: pd.DataFrame = all_data_df
    pred_selected_df: pd.DataFrame = pred_df

    def engagement_slider_changed(cur_engagement_index):
        nonlocal cur_engagement
        cur_engagement = engagements[int(cur_engagement_index)]
        change_engagement_dependent_data()

    def engagement_back_clicked():
        cur_engagement_index = int(engagement_slider.get())
        if cur_engagement_index > 0:
            cur_engagement_index -= 1
            engagement_slider.set(cur_engagement_index)
            engagement_slider_changed(cur_engagement_index)

    def engagement_forward_clicked():
        cur_engagement_index = int(engagement_slider.get())
        if cur_engagement_index < len(engagements) - 1:
            cur_engagement_index += 1
            engagement_slider.set(cur_engagement_index)
            engagement_slider_changed(cur_engagement_index)

    def tick_slider_changed(cur_tick_index_str):
        global cur_tick
        cur_tick_index = int(cur_tick_index_str)
        cur_index = indices[cur_tick_index]
        cur_tick = ticks[cur_tick_index]
        cur_demo_tick = demo_ticks[cur_tick_index]
        cur_game_tick = game_ticks[cur_tick_index]
        tick_id_text_var.set("Tick ID: " + str(cur_tick))
        tick_demo_id_text_var.set("Demo Tick ID: " + str(cur_demo_tick))
        tick_game_id_text_var.set("Game Tick ID: " + str(cur_game_tick))
        input_ax_objs.update_aim_plot(selected_df, cur_tick, canvas, first_hit_view_angle_reference)
        if pred_df is not None:
            pred_ax_objs.update_aim_plot(pred_selected_df, cur_tick, canvas, first_hit_view_angle_reference)
        cur_row = selected_df.loc[cur_index, :]
        columns = first_hit_columns if first_hit_view_angle_reference else cur_hit_columns
        engagement_id_text_var.set(f"Round ID: {int(cur_row.loc['round id'])}, "
                                   f"Engagement ID: {int(cur_engagement)}")
        text_data_text_var.set(f"attacker: {int(cur_row.loc['attacker player id'].item())}, "
                               f"victim: {int(cur_row.loc['victim player id'].item())}, "
                               f"cur view: ({cur_row.loc[columns.cur_x_column].item():.2f}, "
                               f"{cur_row.loc[columns.cur_y_column].item():.2f})")


    def step_back_clicked():
        cur_tick_index = int(tick_slider.get())
        if cur_tick_index > 0:
            cur_tick_index -= 1
            tick_slider.set(cur_tick_index)
            tick_slider_changed(cur_tick_index)


    play_active: bool = False
    num_play_updates_sleeping: int = 0
    def play_clicked():
        nonlocal play_active, num_play_updates_sleeping
        play_active = not play_active
        if play_active:
            play_button.configure(bg='green')
            # technically not sleeping, but need to increment by 1 so -= 1 math works out
            num_play_updates_sleeping += 1
            play_update()
        else:
            play_button.configure(bg=orig_player_button_color)


    def play_update():
        nonlocal num_play_updates_sleeping
        num_play_updates_sleeping -= 1
        if play_active and num_play_updates_sleeping == 0:
            step_forward_clicked()
            num_play_updates_sleeping += 1
            play_button.after(250, play_update)


    def step_forward_clicked():
        cur_tick_index = int(tick_slider.get())
        if cur_tick_index < len(ticks) - 1:
            cur_tick_index += 1
            tick_slider.set(cur_tick_index)
            tick_slider_changed(cur_tick_index)

    first_hit_view_angle_reference = True
    def toggle_reference_clicked():
        nonlocal first_hit_view_angle_reference
        first_hit_view_angle_reference = not first_hit_view_angle_reference
        cur_tick_index = int(tick_slider.get())
        if first_hit_view_angle_reference:
            input_ax.set_title(input_ax.base_title + first_hit_title_suffix)
            if pred_df is not None:
                pred_ax.set_title(pred_ax.base_title + first_hit_title_suffix)
        else:
            input_ax.set_title(input_ax.base_title + cur_pos_title_suffix)
            if pred_df is not None:
                pred_ax.set_title(pred_ax.base_title + cur_pos_title_suffix)
        tick_slider_changed(cur_tick_index)

    # state setters
    def change_engagement_dependent_data():
        nonlocal selected_df, pred_selected_df, cur_engagement, indices, ticks, demo_ticks, game_ticks
        selected_df = all_data_df.loc[all_data_df['engagement id'] == cur_engagement]
        if pred_df is not None:
            pred_selected_df = pred_df.loc[all_data_df['engagement id'] == cur_engagement]

        indices = selected_df.loc[:, 'index'].tolist()
        ticks = selected_df.loc[:, 'tick id'].tolist()
        demo_ticks = selected_df.loc[:, 'demo tick id'].tolist()
        game_ticks = selected_df.loc[:, 'game tick id'].tolist()
        tick_slider.configure(to=len(ticks)-1)
        tick_slider.set(0)
        tick_slider_changed(0)


    s = ttk.Style()
    s.theme_use('alt')
    # creating engagement slider and label
    engagement_id_frame = tk.Frame(window)
    engagement_id_frame.pack(pady=5)
    engagement_id_text_var = tk.StringVar()
    engagement_id_label = tk.Label(engagement_id_frame, textvariable=engagement_id_text_var)
    engagement_id_label.pack(side="left")

    engagement_frame = tk.Frame(window)
    engagement_frame.pack(pady=5)

    engagement_slider = tk.Scale(
        engagement_frame,
        from_=0,
        to=len(engagements)-1,
        orient='horizontal',
        showvalue=0,
        length=500,
        command=engagement_slider_changed
    )
    engagement_slider.pack(side="left")

    # engagegment id stepper
    engagement_step_frame = tk.Frame(window)
    engagement_step_frame.pack(pady=5)
    back_engagement_button = tk.Button(engagement_step_frame, text="⏪", command=engagement_back_clicked)
    back_engagement_button.pack(side="left")
    forward_engagement_button = tk.Button(engagement_step_frame, text="⏩", command=engagement_forward_clicked)
    forward_engagement_button.pack(side="left")

    # creating tick slider and label
    tick_id_frame = tk.Frame(window)
    tick_id_frame.pack(pady=5)

    tick_id_text_var = tk.StringVar()
    tick_id_label = tk.Label(tick_id_frame, textvariable=tick_id_text_var)
    tick_id_label.pack(side="left")

    tick_demo_id_text_var = tk.StringVar()
    tick_demo_id_label = tk.Label(tick_id_frame, textvariable=tick_demo_id_text_var)
    tick_demo_id_label.pack(side="left")

    tick_game_id_text_var = tk.StringVar()
    tick_game_id_label = tk.Label(tick_id_frame, textvariable=tick_game_id_text_var)
    tick_game_id_label.pack(side="left")

    tick_slider_frame = tk.Frame(window)
    tick_slider_frame.pack(pady=5)

    tick_slider = tk.Scale(
        tick_slider_frame,
        from_=0,
        to=100,
        orient='horizontal',
        showvalue=0,
        length=500,
        command=tick_slider_changed
    )
    tick_slider.pack(side="left")

    # creating tick play/pause/step buttons
    tick_step_frame = tk.Frame(window)
    tick_step_frame.pack(pady=5)
    back_step_button = tk.Button(tick_step_frame, text="⏪", command=step_back_clicked)
    back_step_button.pack(side="left")
    play_button = tk.Button(tick_step_frame, text="⏯", command=play_clicked)
    orig_player_button_color = play_button.cget("background")
    play_button.pack(side="left")
    forward_step_button = tk.Button(tick_step_frame, text="⏩", command=step_forward_clicked)
    forward_step_button.pack(side="left")
    frame_of_reference_button = tk.Button(tick_step_frame, text="toggle reference", command=toggle_reference_clicked)
    frame_of_reference_button.pack(side="left")

    # creating text label
    text_data_frame = tk.Frame(window)
    text_data_frame.pack(pady=5)

    text_data_text_var = tk.StringVar()
    text_data_label = tk.Label(text_data_frame, textvariable=text_data_text_var)
    text_data_label.pack(side="left")

    # initial value settings
    engagement_slider_changed(0)
    tick_slider_changed(0)

    # Start the GUI
    window.mainloop()

if __name__ == "__main__":
    all_data_df = pd.read_csv(data_path)
    vis(all_data_df)