from typing import Optional

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure, Axes
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy.typing as npt
import tkinter as tk
from tkinter import ttk

from learn_bot.engagement_aim.dataset import base_abs_x_pos_column, base_abs_y_pos_column
from learn_bot.engagement_aim.output_plotting import filter_df
from learn_bot.engagement_aim.vis.child_window import ChildWindow
from learn_bot.engagement_aim.vis.vis_similar_trajectories import compute_position_difference, default_speed_ticks
from learn_bot.libs.temporal_column_names import get_temporal_field_str

players_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'local_data' / 'players.csv'
players_df = pd.read_csv(players_path, names=["id", "game id", "name", "steam id"], index_col=0)
players_df_name_indexed = players_df.set_index('name')
selected_player_id = 0
players_id_offset = 1

speed_col = "speed (t)"
accel_col = "accel"
speed_ax: Optional[Axes] = None
accel_ax: Optional[Axes] = None


@dataclass
class MouseBins:
    speed_bins: npt.ArrayLike
    accel_bins: npt.ArrayLike


def compute_mouse_movement_bits(all_data_df: pd.DataFrame) -> MouseBins:
    movement_df = all_data_df.copy()
    compute_position_difference(movement_df, base_abs_x_pos_column, base_abs_y_pos_column, speed_col,
                                -1 * default_speed_ticks, 0)
    speed_filtered_movement_df = filter_df(movement_df, speed_col)
    #print(f"{max(movement_df[speed_col])}, {movement_df[speed_col].idxmax()}")
    #max_idx = movement_df[speed_col].idxmax()
    #test_df = movement_df.iloc[[max_idx]].copy()
    #compute_position_difference(test_df, base_abs_x_pos_column, base_abs_y_pos_column, speed_col,
    #                            -1 * default_speed_ticks, 0)
    #print(test_df.to_string())
    #for i in range(-12, 6):
    #    print(f"x {i}: {test_df.loc[:, get_temporal_field_str(base_abs_x_pos_column, i)].item()}, "
    #          f"y {i}: {test_df.loc[:, get_temporal_field_str(base_abs_y_pos_column, i)].item()}")
    _, speed_bins = np.histogram(speed_filtered_movement_df[speed_col].to_numpy(), bins=100)
    next_speed_col = f"speed (t+{default_speed_ticks})"
    compute_position_difference(movement_df, base_abs_x_pos_column, base_abs_y_pos_column, next_speed_col,
                                0, default_speed_ticks)
    movement_df[accel_col] = movement_df[next_speed_col] - movement_df[speed_col]
    accel_filtered_movement_df = filter_df(movement_df, accel_col)
    _, accel_bins = np.histogram(accel_filtered_movement_df[accel_col].to_numpy(), bins=100)
    return MouseBins(speed_bins, accel_bins)


def compute_mouse_movement_distributions(data_df: pd.DataFrame, player_id: int, bins: MouseBins, speed_ax, accel_ax):
    player_data_df = data_df[data_df['attacker player id'] == player_id].copy()
    compute_position_difference(player_data_df, base_abs_x_pos_column, base_abs_y_pos_column, speed_col,
                                -1 * default_speed_ticks, 0)
    next_speed_col = f"speed (t+{default_speed_ticks})"
    compute_position_difference(player_data_df, base_abs_x_pos_column, base_abs_y_pos_column, next_speed_col,
                                0, default_speed_ticks)
    player_data_df[accel_col] = player_data_df[next_speed_col] - player_data_df[speed_col]

    speed_ax.set_title("5-Tick Mouse Speed")
    player_data_df.hist(speed_col, ax=speed_ax, bins=bins.speed_bins,
                        weights=np.ones_like(player_data_df.index) / len(player_data_df.index))
    speed_ax.set_ylim(top=0.55)
    accel_ax.set_title("5-Tick Mouse Acceleration")
    player_data_df.hist(accel_col, ax=accel_ax, bins=bins.accel_bins,
                        weights=np.ones_like(player_data_df.index) / len(player_data_df.index))
    accel_ax.set_ylim(top=0.55)


def update_distribution_plots():
    child_window.figure.suptitle(f"{players_df.loc[selected_player_id, 'name']} Distributions")
    compute_mouse_movement_distributions(data_df, players_df_name_indexed.loc[selected_player_id, 'id'], mouse_bins)

    child_window.canvas.draw()


def check_combo_players(event):
    global selected_player_id
    value = event.widget.get()

    if value == '':
        players_combo_box['values'] = players_df_name_indexed.index.to_list()
    else:
        data = []
        for item in players_df_name_indexed.index:
            if value.lower() in item.lower():
                data.append(item)

        players_combo_box['values'] = data
    if value in players_df_name_indexed.index:
        selected_player_id = players_df_name_indexed.loc[value, 'id']
        players_combo_box.configure(style='Valid.TCombobox')
        update_distribution_plots()
    else:
        players_combo_box.configure(style='Invalid.TCombobox')


def update_selected_players(event):
    global selected_player_id
    selected_player_id = players_df_name_indexed.loc[event.widget.get(), 'id']
    players_combo_box.configure(style='Valid.TCombobox')
    update_distribution_plots()

data_df: Optional[pd.DataFrame] = None
mouse_bins: Optional[MouseBins] = None
child_window = ChildWindow()
players_combo_box: Optional[ttk.Combobox] = None

def plot_distributions(parent_window: tk.Tk, all_data_df: pd.DataFrame):
    global mouse_bins, data_df, selected_player_id, speed_ax, accel_ax
    if child_window.initialize(parent_window, (11, 11)):
        mouse_bins = compute_mouse_movement_bits(all_data_df)
        data_df = all_data_df

        speed_ax, accel_ax = child_window.figure.subplots(nrows=2, ncols=1)

        selected_player_id = players_df.index[0]

        players_combo_box = ttk.Combobox(child_window.window, style='Valid.TCombobox')
        players_combo_box['values'] = players_df_name_indexed.index
        players_combo_box.current(0)
        players_combo_box.bind('<KeyRelease>', check_combo_players)
        players_combo_box.bind("<<ComboboxSelected>>", update_selected_players)
        players_combo_box.pack(pady=5)

        update_distribution_plots()
