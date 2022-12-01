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

from learn_bot.engagement_aim.dataset import base_abs_x_pos_column, base_abs_y_pos_column, base_hit_victim_column, \
    base_ticks_since_last_fire_column, base_relative_x_pos_column, base_recoil_x_column, base_relative_y_pos_column, \
    base_recoil_y_column, base_victim_relative_aabb_max_y, base_victim_relative_aabb_min_y, \
    base_victim_relative_aabb_max_x, base_victim_relative_aabb_min_x
from learn_bot.engagement_aim.output_plotting import filter_df
from learn_bot.engagement_aim.vis.child_window import ChildWindow
from learn_bot.engagement_aim.vis.vis_similar_trajectories import compute_position_difference, default_speed_ticks, \
    normalize_columns
from learn_bot.libs.temporal_column_names import get_temporal_field_str

players_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'local_data' / 'players.csv'
players_df = pd.read_csv(players_path, names=["id", "game id", "name", "steam id"], index_col=0)
player_names = players_df.loc[:, 'name'].unique().to_list()
selected_player = None
players_id_offset = 1

speed_col = "speed (t)"
next_speed_col = f"speed (t+{default_speed_ticks})"
accel_col = "accel"
speed_ax: Optional[Axes] = None
accel_ax: Optional[Axes] = None
hit_ax: Optional[Axes] = None
fire_ax: Optional[Axes] = None


@dataclass
class MovementBins:
    df: pd.DataFrame
    speed_bins: npt.ArrayLike
    accel_bins: npt.ArrayLike


def compute_mouse_movement_bins(data_df: pd.DataFrame) -> MovementBins:
    movement_df = data_df.copy()

    # compute speed
    compute_position_difference(movement_df, base_abs_x_pos_column, base_abs_y_pos_column, speed_col,
                                -1 * default_speed_ticks, 0)
    speed_filtered_movement_df = filter_df(movement_df, speed_col, 0., 0.05)
    _, speed_bins = np.histogram(speed_filtered_movement_df[speed_col].to_numpy(), bins=100)

    # compute accel
    compute_position_difference(movement_df, base_abs_x_pos_column, base_abs_y_pos_column, next_speed_col,
                                0, default_speed_ticks)
    movement_df[accel_col] = movement_df[next_speed_col] - movement_df[speed_col]
    accel_filtered_movement_df = filter_df(movement_df, accel_col, 0.025, 0.025)
    _, accel_bins = np.histogram(accel_filtered_movement_df[accel_col].to_numpy(), bins=100)

    return MovementBins(movement_df, speed_bins, accel_bins)


INVALID_NAME = "invalid"
def compute_mouse_movement_distributions(player_name: str, bins: MovementBins):
    player_data_df = bins.df[bins.df['attacker player name'] == player_name].copy() \
        if player_name != INVALID_NAME else bins.df.copy()

    speed_ax.clear()
    speed_ax.set_title("5-Tick Mouse Speed")
    player_data_df.hist(speed_col, ax=speed_ax, bins=bins.speed_bins,
                        weights=np.ones_like(player_data_df.index) / len(player_data_df.index))
    speed_ax.set_ylim(top=0.55)

    accel_ax.clear()
    accel_ax.set_title("5-Tick Mouse Acceleration")
    player_data_df.hist(accel_col, ax=accel_ax, bins=bins.accel_bins,
                        weights=np.ones_like(player_data_df.index) / len(player_data_df.index))
    accel_ax.set_ylim(top=0.55)


@dataclass
class HitFireBins:
    hit_df: pd.DataFrame
    hit_x_bins: npt.ArrayLike
    hit_y_bins: npt.ArrayLike
    fire_df: pd.DataFrame
    fire_x_bins: npt.ArrayLike
    fire_y_bins: npt.ArrayLike


relative_x_pos_with_recoil_column = "relative x pos with recoil"
relative_y_pos_with_recoil_column = "relative y pos with recoil"
def compute_normalized_with_recoil(data_df: pd.DataFrame):
    data_df[relative_x_pos_with_recoil_column] = \
        data_df[get_temporal_field_str(base_relative_x_pos_column, 0)] + \
        data_df[get_temporal_field_str(base_recoil_x_column, 0)]
    data_df[relative_y_pos_with_recoil_column] = \
        data_df[get_temporal_field_str(base_relative_y_pos_column, 0)] + \
        data_df[get_temporal_field_str(base_recoil_y_column, 0)]

    normalize_columns(data_df, relative_x_pos_with_recoil_column, relative_y_pos_with_recoil_column,
                      get_temporal_field_str(base_victim_relative_aabb_min_x, 0),
                      get_temporal_field_str(base_victim_relative_aabb_min_y, 0),
                      get_temporal_field_str(base_victim_relative_aabb_max_x, 0),
                      get_temporal_field_str(base_victim_relative_aabb_max_y, 0))


def compute_hit_fire_bins(data_df: pd.DataFrame) -> HitFireBins:
    hit_df = data_df.copy()
    hit_df = hit_df[hit_df[get_temporal_field_str(base_hit_victim_column, 0)] == 1]
    compute_normalized_with_recoil(hit_df)

    _, hit_x_bins, hit_y_bins = np.histogram2d(hit_df[relative_x_pos_with_recoil_column].to_numpy(),
                                               hit_df[relative_y_pos_with_recoil_column].to_numpy(),
                                               bins=100)

    fire_df = data_df.copy()
    fire_df = fire_df[fire_df[get_temporal_field_str(base_ticks_since_last_fire_column, 0)] == 0]
    compute_normalized_with_recoil(fire_df)

    _, fire_x_bins, fire_y_bins = np.histogram2d(fire_df[relative_x_pos_with_recoil_column].to_numpy(),
                                                 fire_df[relative_y_pos_with_recoil_column].to_numpy(),
                                                 bins=100)

    return HitFireBins(hit_df, hit_x_bins, hit_y_bins, fire_df, fire_x_bins, fire_y_bins)


def compute_hit_fire_distributions(player_name: str, bins: HitFireBins):
    hit_df = bins.hit_df[bins.hit_df['attacker player name'] == player_name].copy() \
        if player_name != INVALID_NAME else bins.hit_df.copy()

    hit_ax.clear()
    hit_heatmap, _, _ = np.histogram2d(hit_df[relative_x_pos_with_recoil_column],
                                       hit_df[relative_y_pos_with_recoil_column],
                                       bins=[bins.hit_x_bins, bins.hit_y_bins])

    hit_heatmap = hit_heatmap.T

    hit_X, hit_Y = np.meshgrid(bins.hit_x_bins, bins.hit_y_bins)
    hit_im = hit_ax.pcolormesh(hit_X, hit_Y, hit_heatmap)
    child_window.figure.colorbar(hit_im, ax=hit_ax)
    hit_ax.set_title(f"Hit Aim With Recoil Distribution")


    fire_df = bins.fire_df[bins.fire_df['attacker player name'] == player_name].copy() \
        if player_name != INVALID_NAME else bins.fire_df.copy()

    fire_ax.clear()
    fire_heatmap, _, _ = np.histogram2d(fire_df[relative_x_pos_with_recoil_column],
                                        fire_df[relative_y_pos_with_recoil_column],
                                        bins=[bins.fire_x_bins, bins.fire_y_bins])

    fire_heatmap = fire_heatmap.T

    fire_X, fire_Y = np.meshgrid(bins.fire_x_bins, bins.fire_y_bins)
    fire_im = fire_ax.pcolormesh(fire_X, fire_Y, fire_heatmap)
    child_window.figure.colorbar(fire_im, ax=fire_ax)
    fire_ax.set_title(f"Fire Aim With Recoil Distribution")


def update_distribution_plots():
    global speed_ax, accel_ax, hit_ax, fire_ax

    child_window.figure.clear()
    (speed_ax, accel_ax), (hit_ax, fire_ax) = child_window.figure.subplots(nrows=2, ncols=2)
    child_window.figure.suptitle(f"{selected_player} Distributions")

    compute_mouse_movement_distributions(selected_player, movement_bins)
    compute_hit_fire_distributions(selected_player, hit_fire_bins)

    child_window.figure.tight_layout()
    child_window.canvas.draw()


def check_combo_players(event):
    global selected_player
    value = event.widget.get()

    if value == '':
        players_combo_box['values'] = player_names
    else:
        data = []
        for item in player_names:
            if value.lower() in item.lower():
                data.append(item)

        players_combo_box['values'] = data
    if value in player_names:
        selected_player = value
        players_combo_box.configure(style='Valid.TCombobox')
        update_distribution_plots()
    else:
        players_combo_box.configure(style='Invalid.TCombobox')


def update_selected_players(event):
    global selected_player
    selected_player = event.widget.get()
    players_combo_box.configure(style='Valid.TCombobox')
    update_distribution_plots()

data_df: Optional[pd.DataFrame] = None
movement_bins: Optional[MovementBins] = None
hit_fire_bins: Optional[HitFireBins] = None
child_window = ChildWindow()
players_combo_box: Optional[ttk.Combobox] = None

def plot_distributions(parent_window: tk.Tk, all_data_df: pd.DataFrame):
    global players_combo_box, movement_bins, hit_fire_bins, data_df, selected_player
    if child_window.initialize(parent_window, (9, 9)):
        data_df = all_data_df.copy()
        attacker_players_and_ids = players_df.copy()
        attacker_players_and_ids = attacker_players_and_ids.reset_index().loc[:, ['name', 'id']]
        attacker_players_and_ids.rename(columns={'name': 'attacker player name', 'id': 'attacker id'}, inplace=True)
        data_df = data_df.merge(attacker_players_and_ids, left_on='attacker player id', right_on='attacker id')

        movement_bins = compute_mouse_movement_bins(data_df)
        hit_fire_bins = compute_hit_fire_bins(data_df)

        selected_player = INVALID_NAME

        players_combo_box = ttk.Combobox(child_window.window, style='Valid.TCombobox')
        players_combo_box['values'] = player_names
        players_combo_box.current(0)
        players_combo_box.bind('<KeyRelease>', check_combo_players)
        players_combo_box.bind("<<ComboboxSelected>>", update_selected_players)
        players_combo_box.pack(pady=5)

        update_distribution_plots()
