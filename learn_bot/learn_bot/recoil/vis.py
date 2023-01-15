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

weapon_id_column = "weapon id"
recoil_index_column = "recoil index (t)"
x_recoil_column = "scaled recoil angle x (t)"
y_recoil_column = "scaled recoil angle y (t)"
x_vel_column = "attacker vel x (t)"
y_vel_column = "attacker vel y (t)"
z_vel_column = "attacker vel z (t)"


@dataclass
class RecoilPlot:
    fig: plt.Figure
    canvas: FigureCanvasTkAgg

    def plot_recoil_distribution(self, hist_ax: plt.Axes, recoil_df: pd.DataFrame, weapon_id: int,
                                 min_recoil_index: float, max_recoil_index: float, min_speed: float, max_speed: float):
        hist_range = [[-10, 10], [-1, 20]]
        speed_col = \
            np.sqrt((recoil_df[x_vel_column].pow(2) + recoil_df[y_vel_column].pow(2) + recoil_df[z_vel_column].pow(2)))
        conditions = (recoil_df[weapon_id_column] == weapon_id) & \
                     (recoil_df[recoil_index_column] >= min_recoil_index) & \
                     (recoil_df[recoil_index_column] <= max_recoil_index) & \
                     (speed_col >= min_speed) & (speed_col <= max_speed)

        selected_recoil_df = recoil_df[conditions]

        hist_ax.clear()
        recoil_heatmap, recoil_x_bins, recoil_y_bins = np.histogram2d(selected_recoil_df[x_recoil_column].to_numpy(),
                                                                       selected_recoil_df[y_recoil_column].to_numpy(),
                                                                       bins=40, range=hist_range)
        recoil_heatmap = recoil_heatmap.T
        recoil_X, recoil_Y = np.meshgrid(recoil_x_bins, recoil_y_bins)
        recoil_im = hist_ax.pcolormesh(recoil_X, recoil_Y, recoil_heatmap)
        self.fig.colorbar(recoil_im, ax=hist_ax)

        hist_ax.set_title("Scaled Recoil Distribution")
        hist_ax.set_xlabel("X Recoil (deg)")
        hist_ax.set_ylabel("Y Recoil (deg)")
        self.canvas.draw()


def vis(all_data_df: pd.DataFrame):
    recoil_df = all_data_df.loc[:, [weapon_id_column, recoil_index_column, x_recoil_column, y_recoil_column,
                                    x_vel_column, y_vel_column, z_vel_column]]
    weapon_ids = all_data_df.loc[:, weapon_id_column].unique().tolist()
    weapon_names = [weapon_id_to_name[index] for index in weapon_ids]

    #This creates the main window of an application
    window = tk.Tk()
    window.title("Weapon Recoil Explorer")
    window.resizable(width=False, height=False)
    window.configure(background='grey')

    fig = Figure(figsize=(5.5, 5.5), dpi=100)
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

    def update_graph():
        fig.clear()
        hist_ax = fig.add_subplot(1, 1, 1)

        mid_recoil_index = float(mid_recoil_index_selector.get())
        range_recoil_index = float(range_recoil_index_selector.get())
        min_recoil_index = mid_recoil_index - range_recoil_index / 2.
        max_recoil_index = mid_recoil_index + range_recoil_index / 2.

        mid_speed = float(mid_speed_selector.get())
        range_speed = float(range_speed_selector.get())
        min_speed = mid_speed - range_speed / 2.
        max_speed = mid_speed + range_speed / 2.

        recoil_index_text_var.set(f"recoil index mid {mid_recoil_index} range {range_recoil_index},"
                                  f"speed mid {mid_speed} range {range_speed}")
        recoil_plot.plot_recoil_distribution(hist_ax, recoil_df, weapon_name_to_id[weapon_selector_variable.get()],
                                             min_recoil_index, max_recoil_index, min_speed, max_speed)

    discrete_selector_frame = tk.Frame(window)
    discrete_selector_frame.pack(pady=5)

    weapon_selector_variable = tk.StringVar()
    weapon_selector_variable.set(weapon_names[0])  # default value
    weapon_selector = tk.OptionMenu(discrete_selector_frame, weapon_selector_variable, *weapon_names,
                                    command=ignore_arg_update_graph)
    weapon_selector.configure(width=20)
    weapon_selector.pack(side="left")

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
        resolution=0.5,
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
    range_recoil_index_selector.set(30)
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

    update_graph()

    # Start the GUI
    window.mainloop()


if __name__ == "__main__":
    all_data_df = pd.read_csv(data_path)
    vis(all_data_df)
