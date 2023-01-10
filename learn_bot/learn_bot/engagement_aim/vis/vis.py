import pandas as pd

from learn_bot.engagement_aim.vis.distributions import plot_distributions
from learn_bot.engagement_aim.vis.vis_similar_trajectories import SimilarityConstraints, SimilarTrajectory, \
    find_similar_trajectories, plot_similar_trajectories_next_movement
from learn_bot.engagement_aim.vis.vis_ax_objs import PerspectiveColumns, AxObjs
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.engagement_aim.dataset import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import gridspec

weapon_type_to_str = {
    0: "Pistol",
    1: "SMG",
    2: "Heavy",
    3: "AR",
    4: "Sniper",
    5: "Unknown"
}

def vis(all_data_df: pd.DataFrame, pred_df: pd.DataFrame = None):
    all_data_df = all_data_df.sort_values(['engagement id', 'tick id'])
    make_index_column(all_data_df)

    if pred_df is not None:
        pred_df = pred_df.sort_values(['engagement id', 'tick id'])
        make_index_column(pred_df)

    #This creates the main window of an application
    window = tk.Tk()
    window.title("Aim Images")
    window.resizable(width=False, height=False)
    window.configure(background='grey')

    # columns for reading data
    first_tick_columns = PerspectiveColumns(base_changed_offset_coordinates.attacker_x_view_angle,
                                            base_changed_offset_coordinates.attacker_y_view_angle,
                                            base_changed_offset_coordinates.victim_aabb_min_x,
                                            base_changed_offset_coordinates.victim_aabb_min_y,
                                            base_recoil_x_column)
    cur_hit_columns = PerspectiveColumns(base_relative_coordinates.attacker_x_view_angle,
                                         base_relative_coordinates.attacker_y_view_angle,
                                         base_relative_coordinates.victim_aabb_min_x,
                                         base_relative_coordinates.victim_aabb_min_y,
                                         base_recoil_x_column)

    # create axes and their objects
    first_tick_title_suffix = " Relative To First Enemy Head"
    cur_pos_title_suffix = " Relative To Cur Enemy Head"
    first_tick_x_label = "Yaw (deg)"
    cur_pos_x_label = "Yaw Delta (deg)"
    first_tick_y_label = "Pitch (deg)"
    cur_pos_y_label = "Pitch Delta (deg)"
    def setPosAxSettings(ax: plt.Axes, title: str):
        ax.base_title = title + " Pos"
        ax.set_title(title + first_tick_title_suffix)
        ax.set_xlabel(first_tick_x_label)
        ax.set_ylabel(first_tick_y_label)
        ax.set_aspect('equal', adjustable='box')

    def setSpeedAxSettings(ax: plt.Axes, title: str):
        ax.set_title(title + " Speed")
        ax.set_xlabel("Engagement Time (s)")
        ax.set_ylabel("Mouse Speed (deg/tick, 5-window median)")

    if pred_df is None:
        fig = Figure(figsize=(16., 5.5), dpi=100)
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                                 left=0.05, right=0.95, bottom=0.1, top=0.95,
                                 wspace=0.1, hspace=0.2, width_ratios=[1, 1.5])
        input_pos_ax = fig.add_subplot(spec[0])
        input_speed_ax = fig.add_subplot(spec[1])
    else:
        fig = Figure(figsize=(16., 10), dpi=100)
        spec = gridspec.GridSpec(ncols=2, nrows=2,
                                 left=0.05, right=0.95, bottom=0.1, top=0.95,
                                 wspace=0.1, hspace=0.2, width_ratios=[1, 1.5])
        input_pos_ax = fig.add_subplot(spec[0])
        input_speed_ax = fig.add_subplot(spec[1])
        pred_pos_ax = fig.add_subplot(spec[2])
        pred_speed_ax = fig.add_subplot(spec[3])
        setPosAxSettings(pred_pos_ax, "Prediction Aim")
        setSpeedAxSettings(pred_speed_ax, "Prediction Aim")
        pred_ax_objs = AxObjs(True, fig, pred_pos_ax, pred_speed_ax, first_tick_columns, cur_hit_columns)
    setPosAxSettings(input_pos_ax, "Input Aim")
    setSpeedAxSettings(input_speed_ax, "Input Aim")
    input_ax_objs = AxObjs(False, fig, input_pos_ax, input_speed_ax, first_tick_columns, cur_hit_columns)

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
    cur_tick_index: int = -1
    selected_df: pd.DataFrame = all_data_df
    not_selected_df: pd.DataFrame = pd.DataFrame()
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
        nonlocal cur_tick, cur_tick_index
        cur_tick_index = int(cur_tick_index_str)
        cur_index = indices[cur_tick_index]
        cur_tick = ticks[cur_tick_index]
        cur_demo_tick = demo_ticks[cur_tick_index]
        cur_game_tick = game_ticks[cur_tick_index]
        tick_id_text_var.set("Tick ID: " + str(cur_tick))
        tick_demo_id_text_var.set("Demo Tick ID: " + str(cur_demo_tick))
        tick_game_id_text_var.set("Game Tick ID: " + str(cur_game_tick))
        input_ax_objs.update_aim_plot(selected_df, cur_tick, canvas, first_tick_view_angle_reference)
        if pred_df is not None:
            pred_ax_objs.update_aim_plot(pred_selected_df, cur_tick, canvas, first_tick_view_angle_reference)
        cur_row = selected_df.loc[cur_index, :]
        columns = first_tick_columns if first_tick_view_angle_reference else cur_hit_columns
        engagement_id_text_var.set(f"Round ID: {int(cur_row.loc['round id'])}, "
                                   f"Engagement ID: {int(cur_engagement)}")
        text_data_text_var.set(f"attacker: {int(cur_row.loc['attacker player id'].item())}, "
                               f"victim: {int(cur_row.loc['victim player id'].item())}, "
                               f"weapon type: {weapon_type_to_str[int(cur_row.loc['weapon type'].item())]}, "
                               f"attacker view: ({cur_row.loc['attacker view angle x (t)'].item():.2f}, "
                               f"{cur_row.loc['attacker view angle y (t)'].item():.2f}), "
                               f"ideal view: ({cur_row.loc['ideal view angle x (t)'].item():.2f}, "
                               f"{cur_row.loc['ideal view angle y (t)'].item():.2f}), "
                               f"cur view: ({cur_row.loc[columns.cur_view_angle_x_column].item():.2f}, "
                               f"{cur_row.loc[columns.cur_view_angle_y_column].item():.2f}), "
                               f"recoil index: {cur_row.loc['recoil index (t)'].item():.2f}, "
                               f"recoil: ({cur_row.loc[columns.recoil_x_column].item():.2f}, "
                               f"{cur_row.loc[columns.recoil_y_column].item():.2f}), "
                               f"attacker vel: ({cur_row.loc['attacker vel x (t)'].item():.2f}, "
                               f"{cur_row.loc['attacker vel y (t)'].item():.2f}, "
                               f"{cur_row.loc['attacker vel z (t)'].item():.2f}), "
                               f"victim vel: ({cur_row.loc['victim vel x (t)'].item():.2f}, "
                               f"{cur_row.loc['victim vel y (t)'].item():.2f}, "
                               f"{cur_row.loc['victim vel z (t)'].item():.2f}),\n"
                               f"ticks since/until fire: ({cur_row.loc['ticks since last fire (t)'].item():.2f}, "
                               f"{cur_row.loc['ticks until next fire (t)'].item():.2f}), "
                               f"ticks since/until hold attack: ({cur_row.loc['ticks since last holding attack (t)'].item():.2f}, "
                               f"{cur_row.loc['ticks until next holding attack (t)'].item():.2f}), "
                               f"victim visible yet: {cur_row.loc[cur_victim_visible_yet_column]}, "
                               f"holding attack: {cur_row.loc['holding attack (t)']}")

    def step_back_clicked():
        nonlocal cur_tick_index
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
        nonlocal cur_tick_index
        if cur_tick_index < len(ticks) - 1:
            cur_tick_index += 1
            tick_slider.set(cur_tick_index)
            tick_slider_changed(cur_tick_index)

    first_tick_view_angle_reference = True
    def toggle_reference_clicked():
        nonlocal first_tick_view_angle_reference
        first_tick_view_angle_reference = not first_tick_view_angle_reference
        if first_tick_view_angle_reference:
            input_pos_ax.set_title(input_pos_ax.base_title + first_tick_title_suffix)
            input_pos_ax.set_xlabel(first_tick_x_label)
            input_pos_ax.set_ylabel(first_tick_y_label)
            if pred_df is not None:
                pred_pos_ax.set_title(pred_pos_ax.base_title + first_tick_title_suffix)
                pred_pos_ax.set_xlabel(first_tick_x_label)
                pred_pos_ax.set_ylabel(first_tick_y_label)
        else:
            input_pos_ax.set_title(input_pos_ax.base_title + cur_pos_title_suffix)
            input_pos_ax.set_xlabel(cur_pos_x_label)
            input_pos_ax.set_ylabel(cur_pos_y_label)
            if pred_df is not None:
                pred_pos_ax.set_title(pred_pos_ax.base_title + cur_pos_title_suffix)
                pred_pos_ax.set_xlabel(cur_pos_x_label)
                pred_pos_ax.set_ylabel(cur_pos_y_label)
        tick_slider_changed(cur_tick_index)

    similar_trajectories: List[SimilarTrajectory] = []
    def update_similar_trajectories():
        nonlocal similar_trajectories
        similarity_constraint = SimilarityConstraints(
            int(next_move_ticks_entry.get()),
            same_alive_state.get(),
            same_visibility_state.get(),
            float(view_relative_to_enemy_radius_results_entry.get()),
            float(mouse_speed_radius_entry.get()),
            input_ax_objs.first_tick_columns.base_cur_view_angle_x_column,
            input_ax_objs.first_tick_columns.base_cur_view_angle_y_column,
            input_ax_objs.cur_head_columns.base_cur_view_angle_x_column,
            input_ax_objs.cur_head_columns.base_cur_view_angle_y_column
        )
        similar_trajectories = find_similar_trajectories(not_selected_df, selected_df, cur_tick, similarity_constraint)
        plot_similar_trajectories_next_movement(window, not_selected_df, similarity_constraint, similar_trajectories)

    def show_all_data_distributions():
        plot_distributions(window, all_data_df)

    def show_pred_data_distributions():
        plot_distributions(window, pred_df)

    # state setters
    def change_engagement_dependent_data():
        nonlocal selected_df, not_selected_df, pred_selected_df, cur_engagement, indices, ticks, demo_ticks, game_ticks
        selected_df = all_data_df.loc[all_data_df['engagement id'] == cur_engagement]
        not_selected_df = all_data_df.loc[all_data_df['engagement id'] != cur_engagement]
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
    s.configure('Valid.TCombobox', fieldbackground='white')
    s.configure('Invalid.TCombobox', fieldbackground='#cfcfcf')
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
    back_engagement_button = tk.Button(engagement_frame, text="⏪", command=engagement_back_clicked)
    back_engagement_button.pack(side="left")
    forward_engagement_button = tk.Button(engagement_frame, text="⏩", command=engagement_forward_clicked)
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
    back_step_button = tk.Button(tick_slider_frame, text="⏪", command=step_back_clicked)
    back_step_button.pack(side="left")
    play_button = tk.Button(tick_slider_frame, text="⏯", command=play_clicked)
    orig_player_button_color = play_button.cget("background")
    play_button.pack(side="left")
    forward_step_button = tk.Button(tick_slider_frame, text="⏩", command=step_forward_clicked)
    forward_step_button.pack(side="left")
    # not predicting relative cols, so can't toggle in this mode
    if pred_df is None:
        frame_of_reference_button = tk.Button(tick_slider_frame, text="toggle reference", command=toggle_reference_clicked)
        frame_of_reference_button.pack(side="left")

    # creating text label
    text_data_frame = tk.Frame(window)
    text_data_frame.pack(pady=5)

    text_data_text_var = tk.StringVar()
    text_data_label = tk.Label(text_data_frame, textvariable=text_data_text_var)
    text_data_label.pack(side="left")

    # creating similarity selector
    similarity_frame = tk.Frame(window)
    similarity_frame.pack(pady=5)
    plot_all_data_distributions_button = tk.Button(similarity_frame, text="Plot Input Distributions",
                                                   command=show_all_data_distributions)
    plot_all_data_distributions_button.pack(side="left")
    if pred_df is not None:
        plot_pred_distributions_button = tk.Button(similarity_frame, text="Plot Pred Distributions",
                                                   command=show_pred_data_distributions)
        plot_pred_distributions_button.pack(side="left")
    update_similar_trajectories_button = tk.Button(similarity_frame, text="Plot Similar Trajectories",
                                                   command=update_similar_trajectories)
    update_similar_trajectories_button.pack(side="left")

    next_move_ticks_label = tk.Label(similarity_frame, text="Next Move Ticks")
    next_move_ticks_label.pack(side="left")
    next_move_ticks_entry = tk.Entry(similarity_frame, width=5)
    next_move_ticks_entry.pack(side="left")
    next_move_ticks_entry.insert(0, "5")

    same_alive_label = tk.Label(similarity_frame, text="Same Alive")
    same_alive_label.pack(side="left")
    same_alive_state = tk.BooleanVar()
    same_alive_checkbutton = tk.Checkbutton(similarity_frame, variable=same_alive_state)
    same_alive_checkbutton.pack(side="left")
    same_alive_checkbutton.select()

    same_visibility_label = tk.Label(similarity_frame, text="Same Visibility")
    same_visibility_label.pack(side="left")
    same_visibility_state = tk.BooleanVar()
    same_visibility_checkbutton = tk.Checkbutton(similarity_frame, variable=same_visibility_state)
    same_visibility_checkbutton.pack(side="left")
    same_visibility_checkbutton.select()

    view_relative_to_enemy_radius_results_label = tk.Label(similarity_frame, text="View Relative To Enemy Radius")
    view_relative_to_enemy_radius_results_label.pack(side="left")
    view_relative_to_enemy_radius_results_entry = tk.Entry(similarity_frame, width=5)
    view_relative_to_enemy_radius_results_entry.pack(side="left")
    view_relative_to_enemy_radius_results_entry.insert(0, "5.")

    mouse_speed_radius_label = tk.Label(similarity_frame, text="Mouse Speed Radius")
    mouse_speed_radius_label.pack(side="left")
    mouse_speed_radius_entry = tk.Entry(similarity_frame, width=5)
    mouse_speed_radius_entry.pack(side="left")
    mouse_speed_radius_entry.insert(0, "1.")

    # initial value settings
    engagement_slider_changed(0)
    tick_slider_changed(0)

    # Start the GUI
    window.mainloop()

if __name__ == "__main__":
    all_data_df = pd.read_csv(data_path)
    vis(all_data_df)