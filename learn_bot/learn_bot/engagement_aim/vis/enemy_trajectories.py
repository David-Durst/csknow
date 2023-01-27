from pathlib import Path

import pandas as pd
from learn_bot.engagement_aim.column_names import *

from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib as mpl
import matplotlib.cm as cm

from learn_bot.engagement_aim.dataset import data_path

trajectory_path = Path(__file__).parent / ".." / "analysis" / "test_timing_data" / "enemy_trajectories.png"

base_cur_offset_pos_x_column = "cur offset pos x"
base_cur_offset_pos_y_column = "cur offset pos y"
base_cur_offset_pos_z_column = "cur offset pos z"
base_first_offset_pos_x_column = "first offset pos x"
base_first_offset_pos_y_column = "first offset pos y"
base_first_offset_pos_z_column = "first offset pos z"
first_tick_in_engagement_column = 'first tick in engagement'
first_attacker_pos_x_in_engagement_column = 'first attacker pos x in engagement'
first_attacker_pos_y_in_engagement_column = 'first attacker pos y in engagement'
first_attacker_pos_z_in_engagement_column = 'first attacker pos z in engagement'

def vis_2d_trajectories(data_df: pd.DataFrame):
    data_df = data_df.copy()

    data_df.sort_values(['engagement id', 'tick id'], inplace=True)

    data_df[first_tick_in_engagement_column] = data_df.groupby('engagement id')['tick id'].transform('first')
    # get first pos offset
    data_df[first_attacker_pos_x_in_engagement_column] = \
        data_df.groupby('engagement id')[get_temporal_field_str(base_attacker_pos_x, 0)].transform('first')
    data_df[first_attacker_pos_y_in_engagement_column] = \
        data_df.groupby('engagement id')[get_temporal_field_str(base_attacker_pos_y, 0)].transform('first')
    data_df[first_attacker_pos_z_in_engagement_column] = \
        data_df.groupby('engagement id')[get_temporal_field_str(base_attacker_pos_z, 0)].transform('first')
    filtered_df = data_df[(data_df['tick id'] - data_df[first_tick_in_engagement_column]) % 13 == 0].copy()

    new_col_names = []
    new_cols = []
    for i in [0, FUTURE_TICKS]:
        filtered_df[get_temporal_field_str(base_cur_offset_pos_x_column, i)] = \
            filtered_df[get_temporal_field_str(base_victim_pos_x, i)] - \
            filtered_df[get_temporal_field_str(base_attacker_pos_x, i)]
        filtered_df[get_temporal_field_str(base_cur_offset_pos_y_column, i)] = \
            filtered_df[get_temporal_field_str(base_victim_pos_y, i)] - \
            filtered_df[get_temporal_field_str(base_attacker_pos_y, i)]
        filtered_df[get_temporal_field_str(base_cur_offset_pos_z_column, i)] = \
            filtered_df[get_temporal_field_str(base_victim_pos_z, i)] - \
            filtered_df[get_temporal_field_str(base_attacker_pos_z, i)]

        filtered_df[get_temporal_field_str(base_first_offset_pos_x_column, i)] = \
            filtered_df[get_temporal_field_str(base_victim_pos_x, i)] - \
            filtered_df[first_attacker_pos_x_in_engagement_column]
        filtered_df[get_temporal_field_str(base_first_offset_pos_y_column, i)] = \
            filtered_df[get_temporal_field_str(base_victim_pos_y, i)] - \
            filtered_df[first_attacker_pos_y_in_engagement_column]
        filtered_df[get_temporal_field_str(base_first_offset_pos_z_column, i)] = \
            filtered_df[get_temporal_field_str(base_victim_pos_z, i)] - \
            filtered_df[first_attacker_pos_z_in_engagement_column]



    fig = Figure(figsize=(12., 8.), dpi=100)

    cur_ax = fig.add_subplot(1, 2, 1)
    cur_ax.set_title('Enemy Trajectory Relative To Cur Attacker Pos')
    cur_ax.set_xlabel('X Offset To Cur')
    cur_ax.set_ylabel('Y Offset To Cur')
    cur_lines = []
    for _, row in filtered_df.iterrows():
        cur_lines.append((
            (
                row[get_temporal_field_str(base_cur_offset_pos_x_column, 0)],
                row[get_temporal_field_str(base_cur_offset_pos_y_column, 0)]
            ),
            (
                row[get_temporal_field_str(base_cur_offset_pos_x_column, FUTURE_TICKS)],
                row[get_temporal_field_str(base_cur_offset_pos_y_column, FUTURE_TICKS)]
            )
        ))

    cur_lc = LineCollection(cur_lines)
    cur_ax.add_collection(cur_lc)
    cur_ax.autoscale()

    first_ax = fig.add_subplot(1, 2, 2)
    first_ax.set_title('Enemy Trajectory Relative To First Attacker Pos')
    first_ax.set_xlabel('X Offset To First')
    first_ax.set_ylabel('Y Offset To First')
    first_lines = []
    first_colors = []

    color_map = mpl.colormaps['plasma']
    min_z = min(filtered_df[get_temporal_field_str(base_first_offset_pos_z_column, 0)])
    max_z = max(filtered_df[get_temporal_field_str(base_first_offset_pos_z_column, 0)])
    norm = mpl.colors.Normalize(vmin=min_z, vmax=max_z, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=color_map)

    for _, row in filtered_df.iterrows():
        first_lines.append((
            (
                row[get_temporal_field_str(base_first_offset_pos_x_column, 0)],
                row[get_temporal_field_str(base_first_offset_pos_y_column, 0)]
            ),
            (
                row[get_temporal_field_str(base_first_offset_pos_x_column, FUTURE_TICKS)],
                row[get_temporal_field_str(base_first_offset_pos_y_column, FUTURE_TICKS)]
            )
        ))
        first_colors.append(mapper.to_rgba(row[get_temporal_field_str(base_first_offset_pos_z_column, 0)]))

    first_lc = LineCollection(first_lines, colors=first_colors)
    first_ax.add_collection(first_lc)
    first_ax.autoscale()

    fig.savefig(trajectory_path)

if __name__ == "__main__":
    all_data_df = pd.read_csv(data_path)
    vis_2d_trajectories(all_data_df)

