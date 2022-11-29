import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy.typing as npt

from learn_bot.engagement_aim.dataset import base_abs_x_pos_column, base_abs_y_pos_column
from learn_bot.engagement_aim.vis.vis_similar_trajectories import compute_position_difference, default_speed_ticks

players_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'local_data' / 'players.csv'
distributions_output_path = Path(__file__).parent / '..' / 'distributions'
players_id_offset = 1

speed_col = "speed (t)"
accel_col = "accel"


@dataclass
class MouseBins:
    speed_bins: npt.ArrayLike
    accel_bins: npt.ArrayLike


def compute_mouse_movement_bits(all_data_df: pd.DataFrame) -> MouseBins:
    movement_df = all_data_df.copy()
    compute_position_difference(movement_df, base_abs_x_pos_column, base_abs_y_pos_column, speed_col,
                                -1 * default_speed_ticks, 0)
    _, speed_bins = np.histogram(movement_df[speed_col].to_numpy())
    next_speed_col = f"speed (t+{default_speed_ticks})"
    compute_position_difference(movement_df, base_abs_x_pos_column, base_abs_y_pos_column, next_speed_col,
                                0, default_speed_ticks)
    movement_df[accel_col] = movement_df[next_speed_col] - movement_df[speed_col]
    _, accel_bins = np.histogram(movement_df[accel_col].to_numpy())
    return MouseBins(speed_bins, accel_bins)


def compute_mouse_movement_distributions(all_data_df: pd.DataFrame, players_df: pd.DataFrame, player_id: int,
                                         bins: MouseBins):
    player_data_df = all_data_df[all_data_df['attacker player id'] == player_id].copy()
    compute_position_difference(player_data_df, base_abs_x_pos_column, base_abs_y_pos_column, speed_col,
                                -1 * default_speed_ticks, 0)
    next_speed_col = f"speed (t+{default_speed_ticks})"
    compute_position_difference(player_data_df, base_abs_x_pos_column, base_abs_y_pos_column, next_speed_col,
                                0, default_speed_ticks)
    player_data_df[accel_col] = player_data_df[next_speed_col] - player_data_df[speed_col]

    fig = Figure(figsize=(11, 11), dpi=100)
    speed_ax, accel_ax = fig.subplots(nrows=2, ncols=1)

    fig.suptitle(f"{players_df.loc[player_id, 'name']} Speed and Acceleration Distributions")
    speed_ax.set_title("5-Tick Mouse Speed")
    player_data_df.hist(speed_col, ax=speed_ax, bins=bins.speed_bins)
    accel_ax.set_title("5-Tick Mouse Acceleration")
    player_data_df.hist(accel_col, ax=accel_ax, bins=bins.accel_bins)

    fig.savefig(distributions_output_path / f"mouse_dist_{players_df.loc[player_id, 'name']}")


def compute_distributions(all_data_df: pd.DataFrame):
    players_df = pd.read_csv(players_path, names=["id", "game id", "name", "steam id"], index_col=0)
    players_df['id'] = players_df.index
    mouse_bins = compute_mouse_movement_bits(all_data_df)
    for i, player_row in players_df.iterrows():
        compute_mouse_movement_distributions(all_data_df, players_df, player_row['id'], mouse_bins)
    print(players_df)