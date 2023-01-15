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


@dataclass
class RecoilPlot:
    fig: plt.Figure
    canvas: FigureCanvasTkAgg

    def plot_recoil_distribution(self, hist_ax: plt.Axes, recoil_df: pd.DataFrame, weapon_id: int,
                                 min_recoil_index: float, max_recoil_index: float):
        hist_range = [[-10, 10], [-1, 20]]
        selected_recoil_df = recoil_df[(recoil_df[weapon_id_column] == weapon_id) &
                                       (recoil_df[recoil_index_column] >= min_recoil_index) &
                                       (recoil_df[recoil_index_column] <= max_recoil_index)]

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
    recoil_df = all_data_df.loc[:, [weapon_id_column, recoil_index_column, x_recoil_column, y_recoil_column]]
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

    def update_graph():
        fig.clear()
        hist_ax = fig.add_subplot(1, 1, 1)
        recoil_plot.plot_recoil_distribution(hist_ax, recoil_df, weapon_name_to_id[weapon_selector_variable.get()],
                                             float(min_recoil_index_entry.get()), float(max_recoil_index_entry.get()))

    weapon_recoil_index_selector_frame = tk.Frame(window)
    weapon_recoil_index_selector_frame.pack(pady=5)

    weapon_selector_variable = tk.StringVar()
    weapon_selector_variable.set(weapon_names[0])  # default value
    weapon_selector = tk.OptionMenu(weapon_recoil_index_selector_frame, weapon_selector_variable, *weapon_names)
    weapon_selector.pack(side="left")

    min_recoil_index_label = tk.Label(weapon_recoil_index_selector_frame, text="Min Recoil Index")
    min_recoil_index_label.pack(side="left")
    min_recoil_index_entry = tk.Entry(weapon_recoil_index_selector_frame, width=5)
    min_recoil_index_entry.pack(side="left")
    min_recoil_index_entry.insert(0, "0.")

    max_recoil_index_label = tk.Label(weapon_recoil_index_selector_frame, text="Max Recoil Index")
    max_recoil_index_label.pack(side="left")
    max_recoil_index_entry = tk.Entry(weapon_recoil_index_selector_frame, width=5)
    max_recoil_index_entry.pack(side="left")
    max_recoil_index_entry.insert(0, "30.")

    update_button = tk.Button(weapon_recoil_index_selector_frame, text="Update", command=update_graph)
    update_button.pack(side="left")
    update_graph()

    # Start the GUI
    window.mainloop()


if __name__ == "__main__":
    all_data_df = pd.read_csv(data_path)
    vis(all_data_df)
