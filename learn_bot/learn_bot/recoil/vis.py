from enum import Enum

import pandas as pd

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import gridspec
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from learn_bot.recoil.weapon_id_name_conversion import weapon_id_to_name, weapon_name_to_id

data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'engagementAim.csv'
saved_path = Path(__file__).parent / 'saved_dataframe.csv'
recoil_saved_path = Path(__file__).parent / 'saved_recoil_dataframe.csv'

core_id_columns = ["id", "round id", "tick id", "demo tick id", "game tick id", "game time", "engagement id"]

weapon_id_column = "weapon id"
recoil_index_column = "recoil index (t)"
base_x_recoil_column = "scaled recoil angle x"
cur_x_recoil_column = "scaled recoil angle x (t)"
delta_x_recoil_column = "delta scaled recoil angle x"
base_y_recoil_column = "scaled recoil angle y"
cur_y_recoil_column = "scaled recoil angle y (t)"
delta_y_recoil_column = "delta scaled recoil angle y"
x_vel_column = "attacker vel x (t)"
y_vel_column = "attacker vel y (t)"
z_vel_column = "attacker vel z (t)"
attacker_duck_amount_column = "attacker duck amount (t)"
ticks_since_last_fire_column = "ticks since last fire (t)"

duck_options = ["any state", "standing", "crouching"]


class FilteredRecoilData:
    all_cols_df: pd.DataFrame
    recoil_cols_df: pd.DataFrame

    def __init__(self, input_df: pd.DataFrame, weapon_id: int, min_recoil_index: float, max_recoil_index: float,
                 min_speed: float, max_speed: float,
                 min_ticks_since_fire: float, max_ticks_since_fire: float,
                 duck_option: str, delta_ticks: int):
        speed_col = \
            np.sqrt((input_df[x_vel_column].pow(2) + input_df[y_vel_column].pow(2) + input_df[z_vel_column].pow(2)))
        conditions = (input_df[weapon_id_column] == weapon_id) & \
                     (input_df[recoil_index_column] >= min_recoil_index) & \
                     (input_df[recoil_index_column] <= max_recoil_index) & \
                     (speed_col >= min_speed) & (speed_col <= max_speed) & \
                     (input_df[ticks_since_last_fire_column] >= min_ticks_since_fire) & \
                     (input_df[ticks_since_last_fire_column] <= max_ticks_since_fire)

        if duck_option == duck_options[1]:
            conditions = conditions & (input_df[attacker_duck_amount_column] > 0.5)
        elif duck_option == duck_options[2]:
            conditions = conditions & (input_df[attacker_duck_amount_column] <= 0.5)

        self.all_cols_df = input_df[conditions].copy()

        old_x_recoil_column = base_x_recoil_column + f" (t-{delta_ticks})"
        self.all_cols_df[delta_x_recoil_column] = \
            self.all_cols_df[cur_x_recoil_column] - self.all_cols_df[old_x_recoil_column]
        old_y_recoil_column = base_y_recoil_column + f" (t-{delta_ticks})"
        self.all_cols_df[delta_y_recoil_column] = \
            self.all_cols_df[cur_y_recoil_column] - self.all_cols_df[old_y_recoil_column]

        self.recoil_cols_df = \
            self.all_cols_df.loc[:, core_id_columns +
                                    [weapon_id_column, recoil_index_column,
                                     ticks_since_last_fire_column, attacker_duck_amount_column,
                                     cur_x_recoil_column, delta_x_recoil_column, old_x_recoil_column,
                                     cur_y_recoil_column, delta_y_recoil_column, old_y_recoil_column]]


@dataclass
class RecoilPlot:
    fig: plt.Figure
    canvas: FigureCanvasTkAgg

    def plot_recoil_distribution(self, abs_hist_ax: plt.Axes, delta_hist_ax: plt.Axes,
                                 selected_recoil_df: pd.DataFrame):
        abs_hist_range = [[-10, 10], [-1, 20]]
        delta_hist_range = [[-0.75, 0.75], [-0.75, 0.75]]

        # plot abs
        abs_recoil_heatmap, abs_recoil_x_bins, abs_recoil_y_bins = \
            np.histogram2d(selected_recoil_df[cur_x_recoil_column].to_numpy(),
                           selected_recoil_df[cur_y_recoil_column].to_numpy(),
                           bins=41, range=abs_hist_range)
        abs_recoil_heatmap = abs_recoil_heatmap.T
        abs_recoil_X, abs_recoil_Y = np.meshgrid(abs_recoil_x_bins, abs_recoil_y_bins)
        abs_recoil_im = abs_hist_ax.pcolormesh(abs_recoil_X, abs_recoil_Y, abs_recoil_heatmap)
        self.fig.colorbar(abs_recoil_im, ax=abs_hist_ax)

        abs_hist_ax.set_title("Absolute Scaled Recoil")
        abs_hist_ax.set_xlabel("X Recoil (deg)")
        abs_hist_ax.set_ylabel("Y Recoil (deg)")
        abs_hist_ax.invert_xaxis()

        # plot delta
        delta_recoil_heatmap, delta_recoil_x_bins, delta_recoil_y_bins = \
            np.histogram2d(selected_recoil_df[delta_x_recoil_column].to_numpy(),
                           selected_recoil_df[delta_y_recoil_column].to_numpy(),
                           bins=41, range=delta_hist_range)
        delta_recoil_heatmap = delta_recoil_heatmap.T
        delta_recoil_X, delta_recoil_Y = np.meshgrid(delta_recoil_x_bins, delta_recoil_y_bins)
        delta_recoil_im = delta_hist_ax.pcolormesh(delta_recoil_X, delta_recoil_Y, delta_recoil_heatmap)
        self.fig.colorbar(delta_recoil_im, ax=delta_hist_ax)

        delta_hist_ax.set_title("Delta Scaled Recoil")
        delta_hist_ax.set_xlabel("Delta X Recoil (deg)")
        delta_hist_ax.set_ylabel("Delta Y Recoil (deg)")
        delta_hist_ax.invert_xaxis()

        self.fig.tight_layout()
        self.canvas.draw()


def vis(recoil_df: pd.DataFrame):
    weapon_ids = all_data_df.loc[:, weapon_id_column].unique().tolist()
    weapon_names = [weapon_id_to_name[index] for index in weapon_ids]
    weapon_names = sorted(weapon_names)

    #This creates the main window of an application
    window = tk.Tk()
    window.title("Weapon Recoil Explorer")
    window.resizable(width=False, height=False)
    window.configure(background='grey')

    fig = Figure(figsize=(12., 5.5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea.
    recoil_plot = RecoilPlot(fig, canvas)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def ignore_arg_update_graph(ignore_arg):
        update_graph()

    last_filtered_recoil_data: FilteredRecoilData

    def update_graph():
        nonlocal last_filtered_recoil_data
        fig.clear()
        abs_hist_ax = fig.add_subplot(1, 2, 1)
        delta_hist_ax = fig.add_subplot(1, 2, 2)

        mid_recoil_index = float(mid_recoil_index_selector.get())
        range_recoil_index = float(range_recoil_index_selector.get())
        min_recoil_index = mid_recoil_index - range_recoil_index / 2.
        max_recoil_index = mid_recoil_index + range_recoil_index / 2.

        mid_speed = float(mid_speed_selector.get())
        range_speed = float(range_speed_selector.get())
        min_speed = mid_speed - range_speed / 2.
        max_speed = mid_speed + range_speed / 2.

        mid_ticks_since_fire = float(mid_ticks_since_fire_selector.get())
        range_ticks_since_fire = float(range_ticks_since_fire_selector.get())
        min_ticks_since_fire = mid_ticks_since_fire - range_ticks_since_fire / 2.
        max_ticks_since_fire = mid_ticks_since_fire + range_ticks_since_fire / 2.

        last_filtered_recoil_data = \
            FilteredRecoilData(recoil_df, weapon_name_to_id[weapon_selector_variable.get()],
                               min_recoil_index, max_recoil_index, min_speed, max_speed,
                               min_ticks_since_fire, max_ticks_since_fire,
                               duck_selector_variable.get(), int(delta_ticks_selector.get()))

        recoil_index_text_var.set(f"recoil index mid {mid_recoil_index} range {range_recoil_index},"
                                  f"speed mid {mid_speed} range {range_speed},"
                                  f"ticks since fire mid {mid_ticks_since_fire} range {range_ticks_since_fire},"
                                  f"delta ticks {int(delta_ticks_selector.get())}")

        recoil_plot.plot_recoil_distribution(abs_hist_ax, delta_hist_ax, last_filtered_recoil_data.all_cols_df)

    def save_graph_data():
        last_filtered_recoil_data.all_cols_df.to_csv(saved_path)
        
    def save_graph_data_recoil_cols():
        last_filtered_recoil_data.recoil_cols_df.to_csv(recoil_saved_path)


    discrete_selector_frame = tk.Frame(window)
    discrete_selector_frame.pack(pady=5)

    weapon_selector_variable = tk.StringVar()
    weapon_selector_variable.set(weapon_names[0])  # default value
    weapon_selector = tk.OptionMenu(discrete_selector_frame, weapon_selector_variable, *weapon_names,
                                    command=ignore_arg_update_graph)
    weapon_selector.configure(width=20)
    weapon_selector.pack(side="left")

    duck_label = tk.Label(discrete_selector_frame, text="Duck Options")
    duck_label.pack(side="left")
    duck_selector_variable = tk.StringVar()
    duck_selector_variable.set(duck_options[0])  # default value
    duck_selector = tk.OptionMenu(discrete_selector_frame, duck_selector_variable, *duck_options,
                                    command=ignore_arg_update_graph)
    duck_selector.configure(width=20)
    duck_selector.pack(side="left")

    save_all_cols_button = tk.Button(discrete_selector_frame, text="Save All Cols", command=save_graph_data)
    save_all_cols_button.pack(side="left")

    save_recoil_cols_button = tk.Button(discrete_selector_frame, text="Save Recoil Cols",
                                        command=save_graph_data_recoil_cols)
    save_recoil_cols_button.pack(side="left")

    recoil_index_text_frame = tk.Frame(window)
    recoil_index_text_frame.pack(pady=5)
    recoil_index_text_var = tk.StringVar()
    recoil_index_label = tk.Label(recoil_index_text_frame, textvariable=recoil_index_text_var)
    recoil_index_label.pack(side="left")

    mid_recoil_index_frame = tk.Frame(window)
    mid_recoil_index_frame.pack(pady=5)
    mid_recoil_index_label = tk.Label(mid_recoil_index_frame, text="Recoil Index Mid")
    mid_recoil_index_label.pack(side="left")
    mid_recoil_index_selector = tk.Scale(
        mid_recoil_index_frame,
        from_=0,
        to=30,
        orient='horizontal',
        showvalue=0,
        length=300,
        command=ignore_arg_update_graph
    )
    mid_recoil_index_selector.pack()

    range_recoil_index_frame = tk.Frame(window)
    range_recoil_index_frame.pack(pady=5)
    range_recoil_index_label = tk.Label(range_recoil_index_frame, text="Recoil Index Range")
    range_recoil_index_label.pack(side="left")
    range_recoil_index_selector = tk.Scale(
        range_recoil_index_frame,
        from_=1,
        to=30,
        orient='horizontal',
        showvalue=0,
        length=300,
        command=ignore_arg_update_graph
    )
    #range_recoil_index_selector.set(30)
    range_recoil_index_selector.pack(side="left")

    mid_speed_frame = tk.Frame(window)
    mid_speed_frame.pack(pady=5)
    mid_speed_label = tk.Label(mid_speed_frame, text="Attacker Speed Mid")
    mid_speed_label.pack(side="left")
    mid_speed_selector = tk.Scale(
        mid_speed_frame,
        from_=0,
        to=250,
        orient='horizontal',
        showvalue=0,
        length=300,
        resolution=0.5,
        command=ignore_arg_update_graph
    )
    mid_speed_selector.pack(side="left")

    range_speed_frame = tk.Frame(window)
    range_speed_frame.pack(pady=5)
    range_speed_label = tk.Label(range_speed_frame, text="Attacker Speed Range")
    range_speed_label.pack(side="left")
    range_speed_selector = tk.Scale(
        range_speed_frame,
        from_=1,
        to=500,
        orient='horizontal',
        showvalue=0,
        length=300,
        command=ignore_arg_update_graph
    )
    range_speed_selector.set(500)
    range_speed_selector.pack(side="left")


    mid_ticks_since_fire_frame = tk.Frame(window)
    mid_ticks_since_fire_frame.pack(pady=5)
    mid_ticks_since_fire_label = tk.Label(mid_ticks_since_fire_frame, text="Ticks Since Fire Mid")
    mid_ticks_since_fire_label.pack(side="left")
    mid_ticks_since_fire_selector = tk.Scale(
        mid_ticks_since_fire_frame,
        from_=0,
        to=100,
        orient='horizontal',
        showvalue=0,
        length=300,
        command=ignore_arg_update_graph
    )
    mid_ticks_since_fire_selector.pack(side="left")

    range_ticks_since_fire_frame = tk.Frame(window)
    range_ticks_since_fire_frame.pack(pady=5)
    range_ticks_since_fire_label = tk.Label(range_ticks_since_fire_frame, text="Ticks Since Fire Range")
    range_ticks_since_fire_label.pack(side="left")
    range_ticks_since_fire_selector = tk.Scale(
        range_ticks_since_fire_frame,
        from_=1,
        to=200,
        orient='horizontal',
        showvalue=0,
        length=300,
        command=ignore_arg_update_graph
    )
    #range_ticks_since_fire_selector.set(200)
    range_ticks_since_fire_selector.pack(side="left")

    delta_ticks_frame = tk.Frame(window)
    delta_ticks_frame.pack(pady=5)
    delta_ticks_label = tk.Label(delta_ticks_frame, text="Delta Ticks")
    delta_ticks_label.pack(side="left")
    delta_ticks_selector = tk.Scale(
        delta_ticks_frame,
        from_=1,
        to=13,
        orient='horizontal',
        showvalue=0,
        length=300,
        command=ignore_arg_update_graph
    )
    delta_ticks_selector.pack(side="left")

    update_graph()

    # Start the GUI
    window.mainloop()


if __name__ == "__main__":
    all_data_df = pd.read_csv(data_path)
    vis(all_data_df)
