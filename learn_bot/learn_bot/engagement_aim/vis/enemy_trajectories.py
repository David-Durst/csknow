from pathlib import Path

import pandas as pd
from matplotlib import gridspec

from learn_bot.engagement_aim.column_names import *

from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib as mpl
import matplotlib.cm as cm

from learn_bot.engagement_aim.dataset import data_path, manual_data_path

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



    fig = Figure(figsize=(25., 16.), dpi=100)

    gs = gridspec.GridSpec(ncols=3, nrows=2,
                           left=0.05, right=0.95, wspace=0.2)
    #left=0.05, right=0.95,
    #wspace=0.1, hspace=0.1, width_ratios=[1, 1.5])

    # cur ax
    cur_ax = fig.add_subplot(gs[0, 0])
    cur_ax.set_title('Enemy Trajectory Relative To Cur Attacker Pos')
    cur_ax.set_xlabel('X Offset To Cur')
    cur_ax.set_ylabel('Y Offset To Cur')
    cur_lines = []
    cur_colors = []

    color_map = mpl.colormaps['plasma']
    min_z = min(filtered_df[get_temporal_field_str(base_cur_offset_pos_z_column, 0)])
    max_z = max(filtered_df[get_temporal_field_str(base_cur_offset_pos_z_column, 0)])
    norm = mpl.colors.Normalize(vmin=min_z, vmax=max_z, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=color_map)

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
        cur_colors.append(mapper.to_rgba(row[get_temporal_field_str(base_cur_offset_pos_z_column, 0)]))

    cur_lc = LineCollection(cur_lines, colors=cur_colors)
    cur_ax.add_collection(cur_lc)
    cur_ax.autoscale()
    cur_xs = cur_ax.get_xlim()
    cur_ys = cur_ax.get_ylim()
    cur_ax.set_xlim(min(cur_xs[0], cur_ys[0]), max(cur_xs[1], cur_ys[1]))
    cur_ax.set_ylim(min(cur_xs[0], cur_ys[0]), max(cur_xs[1], cur_ys[1]))
    cbar = fig.colorbar(mapper)
    cbar.set_label('Enemy Height Relative To Cur Attacker Pos', rotation=270, labelpad=10)

    # first ax
    first_ax = fig.add_subplot(gs[0, 1])
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
    first_xs = first_ax.get_xlim()
    first_ys = first_ax.get_ylim()
    first_ax.set_xlim(min(first_xs[0], first_ys[0]), max(first_xs[1], first_ys[1]))
    first_ax.set_ylim(min(first_xs[0], first_ys[0]), max(first_xs[1], first_ys[1]))
    cbar = fig.colorbar(mapper)
    cbar.set_label('Enemy Height Relative To First Attacker Pos', rotation=270, labelpad=10)

    hist_ax = fig.add_subplot(gs[0, 2])
    filtered_df.hist(get_temporal_field_str(base_first_offset_pos_z_column, 0), bins=20, ax=hist_ax)
    hist_ax.set_title('Enemy Height Relative To First Attacker Pos')
    hist_ax.set_xlabel('Z Offset To First')
    hist_ax.set_ylabel('Num Points')

    view_color = [(0., 0., 1., 0.1)]

    # enemy head ax
    enemy_head_ax = fig.add_subplot(gs[1, 0])
    enemy_head_ax.set_title('Victim Head View Angle')
    enemy_head_ax.set_xlabel('Yaw')
    enemy_head_ax.set_ylabel('Pitch')
    enemy_head_lines = []

    for _, row in filtered_df.iterrows():
        enemy_head_lines.append((
            (
                row[get_temporal_field_str(base_changed_offset_coordinates.victim_aabb_head_x, 0)],
                row[get_temporal_field_str(base_changed_offset_coordinates.victim_aabb_head_y, 0)]
            ),
            (
                row[get_temporal_field_str(base_changed_offset_coordinates.victim_aabb_head_x, FUTURE_TICKS)],
                row[get_temporal_field_str(base_changed_offset_coordinates.victim_aabb_head_y, FUTURE_TICKS)]
            )
        ))

    enemy_head_lc = LineCollection(enemy_head_lines, colors=view_color)
    enemy_head_ax.add_collection(enemy_head_lc)
    enemy_head_ax.autoscale()
    enemy_head_xs = enemy_head_ax.get_xlim()
    enemy_head_ys = enemy_head_ax.get_ylim()
    enemy_head_ax.set_xlim(max(enemy_head_xs[1], enemy_head_ys[1]), min(enemy_head_xs[0], enemy_head_ys[0]))
    enemy_head_ax.set_ylim(max(enemy_head_xs[1], enemy_head_ys[1]), min(enemy_head_xs[0], enemy_head_ys[0]))

    # crosshair ax
    crosshair_ax = fig.add_subplot(gs[1, 1])
    crosshair_ax.set_title('Attacker View Angle')
    crosshair_ax.set_xlabel('Yaw')
    crosshair_ax.set_ylabel('Pitch')
    crosshair_lines = []

    for _, row in filtered_df.iterrows():
        crosshair_lines.append((
            (
                row[get_temporal_field_str(base_changed_offset_coordinates.attacker_x_view_angle, 0)],
                row[get_temporal_field_str(base_changed_offset_coordinates.attacker_y_view_angle, 0)]
            ),
            (
                row[get_temporal_field_str(base_changed_offset_coordinates.attacker_x_view_angle, FUTURE_TICKS)],
                row[get_temporal_field_str(base_changed_offset_coordinates.attacker_y_view_angle, FUTURE_TICKS)]
            )
        ))

    crosshair_lc = LineCollection(crosshair_lines, colors=view_color)
    crosshair_ax.add_collection(crosshair_lc)
    crosshair_ax.autoscale()
    crosshair_xs = crosshair_ax.get_xlim()
    crosshair_ys = crosshair_ax.get_ylim()
    crosshair_ax.set_xlim(max(crosshair_xs[1], crosshair_ys[1]), min(crosshair_xs[0], crosshair_ys[0]))
    crosshair_ax.set_ylim(max(crosshair_xs[1], crosshair_ys[1]), min(crosshair_xs[0], crosshair_ys[0]))

    # relative crosshair ax
    relative_crosshair_ax = fig.add_subplot(gs[1, 2])
    relative_crosshair_ax.set_title('Relative View Angle')
    relative_crosshair_ax.set_xlabel('Yaw')
    relative_crosshair_ax.set_ylabel('Pitch')
    relative_crosshair_lines = []

    for _, row in filtered_df.iterrows():
        relative_crosshair_lines.append((
            (
                row[get_temporal_field_str(base_relative_coordinates.attacker_x_view_angle, 0)],
                row[get_temporal_field_str(base_relative_coordinates.attacker_y_view_angle, 0)]
            ),
            (
                row[get_temporal_field_str(base_relative_coordinates.attacker_x_view_angle, FUTURE_TICKS)],
                row[get_temporal_field_str(base_relative_coordinates.attacker_y_view_angle, FUTURE_TICKS)]
            )
        ))

    relative_crosshair_lc = LineCollection(relative_crosshair_lines, colors=view_color)
    relative_crosshair_ax.add_collection(relative_crosshair_lc)
    relative_crosshair_ax.autoscale()
    relative_crosshair_xs = relative_crosshair_ax.get_xlim()
    relative_crosshair_ys = relative_crosshair_ax.get_ylim()
    relative_crosshair_ax.set_xlim(max(relative_crosshair_xs[1], relative_crosshair_ys[1]),
                                   min(relative_crosshair_xs[0], relative_crosshair_ys[0]))
    relative_crosshair_ax.set_ylim(max(relative_crosshair_xs[1], relative_crosshair_ys[1]),
                                   min(relative_crosshair_xs[0], relative_crosshair_ys[0]))

    fig.savefig(trajectory_path)


orig_dataset = True
if __name__ == "__main__":
    if orig_dataset:
        all_data_df = pd.read_csv(data_path)
    else:
        all_data_df = pd.read_csv(manual_data_path)
    vis_2d_trajectories(all_data_df)

