import pandas as pd
from pathlib import Path

from learn_bot.libs.df_grouping import make_index_column
from learn_bot.engagement_aim.dataset import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

def vis(all_data_df: pd.DataFrame):
    # transform input and output
    column_transformers = IOColumnTransformers(input_column_types, output_column_types, all_data_df)

    all_data_df = all_data_df.sort_values(['engagement id', 'tick id'])
    make_index_column(all_data_df)
    engagement_start_ends = all_data_df.groupby('engagement id').agg(
        engagement_id=('engagement id', 'first'),
        start_index=('index', 'first'),
        end_index=('index', 'last'),
        round_id=('round id', 'first'))

    #This creates the main window of an application
    window = tk.Tk()
    window.title("Aim Images")
    window.geometry("700x780")
    window.configure(background='grey')

    # cur
    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot()
    all_line = None
    prior_line = None
    prior_future_x_column = temporal_io_float_column_names.vis_columns[0]
    prior_future_y_column = temporal_io_float_column_names.vis_columns[1]
    present_line = None
    present_series_line = None
    present_x_columns = temporal_io_float_column_names.get_matching_cols(base_float_columns[0])
    present_y_columns = temporal_io_float_column_names.get_matching_cols(base_float_columns[1])
    future_line = None
    ax.set_xlabel("Yaw Delta (deg / target height deg)")
    ax.set_ylabel("Pitch Delta (deg / target height deg)")
    ax.set_aspect('equal', adjustable='box')
    ax.invert_xaxis()

    canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_aim_plot(data_df, tick_id):
        nonlocal all_line, prior_line, present_line, present_series_line, future_line

        prior_df = data_df[data_df['tick id'] < tick_id]
        prior_x_np = prior_df.loc[:, prior_future_x_column].to_numpy()
        prior_y_np = prior_df.loc[:, prior_future_y_column].to_numpy()

        present_df = data_df[data_df['tick id'] == tick_id]
        present_x_np = present_df.loc[:, prior_future_x_column].to_numpy()
        present_y_np = present_df.loc[:, prior_future_y_column].to_numpy()

        # shows series used in prediction, not sust single point
        present_series_df = data_df[data_df['tick id'] == tick_id].iloc[0, :]
        present_series_x_np = present_series_df.loc[present_x_columns].to_numpy()
        present_series_y_np = present_series_df.loc[present_y_columns].to_numpy()

        future_df = data_df[data_df['tick id'] > tick_id]
        future_x_np = future_df.loc[:, prior_future_x_column].to_numpy()
        future_y_np = future_df.loc[:, prior_future_y_column].to_numpy()

        all_x_np = data_df.loc[:, prior_future_x_column].to_numpy()
        all_y_np = data_df.loc[:, prior_future_y_column].to_numpy()

        if prior_line is None:
            # ax1.plot(x, y,color='#FF0000', linewidth=2.2, label='Example line',
            #           marker='o', mfc='black', mec='black', ms=10)
            line_gray = (0.87, 0.87, 0.87, 1)
            all_line, = ax.plot(all_x_np, all_y_np, color=line_gray, label="_nolegend_")
            present_series_yellow = "#A89932FF"
            present_series_line, = ax.plot(present_series_x_np, present_series_y_np,
                                           linestyle="None", label="Model Feature",
                                           marker='o', mfc="None", mec=present_series_yellow,
                                           markersize=10)
            prior_blue = "#00D5FAFF"
            prior_line, = ax.plot(prior_x_np, prior_y_np, linestyle="None", label="Past",
                                  marker='o', mfc=prior_blue, mec=prior_blue)
            future_gray = "#727272FF"
            future_line, = ax.plot(future_x_np, future_y_np, linestyle="None", label="Future",
                                   marker='o', mfc=future_gray, mec=future_gray)
            present_red = (1., 0., 0., 0.5)
            present_line, = ax.plot(present_x_np, present_y_np, linestyle="None", label="Present",
                                    marker='o', mfc=present_red, mec=present_red)
        else:
            all_line.set_data(all_x_np, all_y_np)
            present_series_line.set_data(present_series_x_np, present_series_y_np)
            prior_line.set_data(prior_x_np, prior_y_np)
            future_line.set_data(future_x_np, future_y_np)
            present_line.set_data(present_x_np, present_y_np)

        # recompute the ax.dataLim
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale()
        ax.legend()
        xmax, xmin = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lim_min = min([xmin, ymin])
        lim_max = max([xmax, ymax])
        # inverted xaxis so need to flip
        ax.set_xlim(lim_max, lim_min)
        ax.set_ylim(lim_min, lim_max)

        # required to update canvas and attached toolbar!
        canvas.draw()

    engagements = all_data_df.loc[:, 'engagement id'].unique().tolist()
    indices = []
    ticks = []
    demo_ticks = []
    game_ticks = []
    cur_engagement: int = -1
    cur_tick: int = -1
    selected_df: pd.DataFrame = all_data_df


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
        update_aim_plot(selected_df, cur_tick)
        cur_row = selected_df.loc[cur_index, :]
        engagement_id_text_var.set(f"Round ID: {int(cur_row.loc['round id'])}, "
                                   f"Engagement ID: {cur_engagement}")
        text_data_text_var.set(f"attacker: {int(cur_row.loc['attacker player id'].item())}, "
                               f"victim: {int(cur_row.loc['victim player id'].item())}, "
                               f"cur view: ({cur_row.loc[prior_future_x_column].item():.2f}, "
                               f"{cur_row.loc[prior_future_y_column].item():.2f})")


    def step_back_clicked():
        cur_tick_index = int(tick_slider.get())
        if cur_tick_index > 0:
            cur_tick_index -= 1
            tick_slider.set(cur_tick_index)
            tick_slider_changed(cur_tick_index)


    play_active: bool = False
    num_play_updates_sleeping: int = 0
    def play_clicked():
        global play_active, num_play_updates_sleeping
        play_active = not play_active
        if play_active:
            play_button.configure(bg='green')
            # technically not sleeping, but need to increment by 1 so -= 1 math works out
            num_play_updates_sleeping += 1
            play_update()
        else:
            play_button.configure(bg=orig_player_button_color)


    def play_update():
        global num_play_updates_sleeping
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


    # state setters
    def change_engagement_dependent_data():
        nonlocal selected_df, cur_engagement, indices, ticks, demo_ticks, game_ticks
        selected_df = all_data_df.loc[all_data_df['engagement id'] == cur_engagement]

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