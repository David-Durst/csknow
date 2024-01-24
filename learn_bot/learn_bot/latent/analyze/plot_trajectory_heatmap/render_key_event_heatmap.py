import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.plot_trajectory_heatmap.title_rename_dict import title_rename_dict
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import title_to_team_to_pos_dict
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, convert_to_canvas_coordinates
from learn_bot.latent.transformer_nested_hidden_latent_model import d2_min, d2_max

fig_length = 10

title_to_num_y_points_less_than_1000: Dict[str, int] = {}
title_to_num_x_points_between_negative_1200_and_0: Dict[str, int] = {}
title_to_num_points: Dict[str, int] = {}

def clear_event_counters():
    global title_to_num_x_points_between_negative_1200_and_0, title_to_num_y_points_less_than_1000, \
        title_to_num_points
    title_to_num_y_points_less_than_1000 = {}
    title_to_num_x_points_between_negative_1200_and_0 = {}
    title_to_num_points = {}

def create_heatmap(title: str, ct_team: bool, title_to_team_to_heatmap: Dict[str, Dict],
                   x_pos: List[float], y_pos: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    heatmap, x_bins, y_bins = np.histogram2d(x_pos, y_pos, bins=25,
                                             range=[[d2_min[0], d2_max[0]], [d2_min[1], d2_max[1]]])
    if title not in title_to_num_y_points_less_than_1000:
        title_to_num_y_points_less_than_1000[title] = 0
        title_to_num_x_points_between_negative_1200_and_0[title] = 0
        title_to_num_points[title] = 0
    title_to_num_y_points_less_than_1000[title] += len([y for y in y_pos if y < 1000])
    title_to_num_x_points_between_negative_1200_and_0[title] += len([x for x in x_pos if x > -1200 and x < 0])
    title_to_num_points[title] += len(x_pos)
    heatmap = heatmap.T
    if title not in title_to_team_to_heatmap:
        title_to_team_to_heatmap[title] = {}
    title_to_team_to_heatmap[title][ct_team] = heatmap
    return x_bins, y_bins


def plot_heatmap(title: str, event: str, ct_team: bool, title_to_team_to_heatmap: Dict[str, Dict],
                 x_pos: List[float], max_bin_value: int,
                 x_bins: np.ndarray, y_bins: np.ndarray, fig: plt.Figure, ax: plt.Axes, cmap, first_row: bool, first_col: bool):
    if first_row:
        ax.set_title(title_rename_dict[title], fontsize=8)
    if first_col:
        ax.set_ylabel("Offense" if ct_team else "Defense", fontsize=8)
    grid_x, grid_y = np.meshgrid(x_bins, y_bins)
    heatmap_im = ax.pcolor(grid_x, grid_y, title_to_team_to_heatmap[title][ct_team], vmin=1,
                               vmax=max_bin_value, cmap=cmap)#, #linewidth=0, alpha=1, rasterized=True)
    heatmap_im.set_edgecolor('face')
    #cbar = fig.colorbar(heatmap_im, ax=ax)
    #cbar.ax.set_ylabel(f'Number of {event}', rotation=270, labelpad=15, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    return heatmap_im


def plot_key_event_heatmaps(title_to_team_to_key_event_pos: title_to_team_to_pos_dict,
                            trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    if not trajectory_filter_options.filtering_key_events():
        return

    clear_event_counters()

    num_titles = len(title_to_team_to_key_event_pos.keys())
    # mul cols by 2 as need ct and t plots
    local_fig_width = 7
    # fig_length * (num_titles + 0.3)
    fig = plt.figure(figsize=(7, 7 * 0.4), constrained_layout=True)
    axs = fig.subplots(2, num_titles, squeeze=False)

    canvas_min_max = convert_to_canvas_coordinates(pd.Series([d2_min[0], d2_max[0]]),
                                                   pd.Series([d2_min[1], d2_max[1]]))
    cropped_d2_img = d2_img.copy().convert("RGBA").crop([min(canvas_min_max[0]), min(canvas_min_max[1]),
                                                         max(canvas_min_max[0]), max(canvas_min_max[1])])
    cropped_np_d2_img = np.asarray(cropped_d2_img)

    cmap = plt.get_cmap('tab20b').copy()
    cmap.set_under((1, 1, 1, 0))

    max_points = 0
    for _, team_to_key_event_pos in title_to_team_to_key_event_pos.items():
        max_points = max(max_points, len(team_to_key_event_pos[True][0]))
        max_points = max(max_points, len(team_to_key_event_pos[False][0]))

    event = ""
    if trajectory_filter_options.only_kill:
        event = "kills"
    elif trajectory_filter_options.only_killed:
        event = "deaths"
    elif trajectory_filter_options.only_killed_or_end:
        event = "deaths_and_ends"
    elif trajectory_filter_options.only_shots:
        event = "shots"

    title_to_team_to_heatmap = {}
    max_bin_value = 0
    for i, (title, team_to_key_event_pos) in enumerate(title_to_team_to_key_event_pos.items()):
        x_bins, y_bins = create_heatmap(title, True, title_to_team_to_heatmap,
                                        team_to_key_event_pos[True][0], team_to_key_event_pos[True][1])
        max_bin_value = max(max_bin_value, title_to_team_to_heatmap[title][True].max())
        create_heatmap(title, False, title_to_team_to_heatmap,
                       team_to_key_event_pos[False][0], team_to_key_event_pos[False][1])
        max_bin_value = max(max_bin_value, title_to_team_to_heatmap[title][False].max())

    for i, (title, team_to_key_event_pos) in enumerate(title_to_team_to_key_event_pos.items()):
        axs[0, i].imshow(cropped_np_d2_img, extent=[d2_min[0], d2_max[0], d2_min[1], d2_max[1]])
        plot_heatmap(title, event, True, title_to_team_to_heatmap,
                     team_to_key_event_pos[True][0], max_bin_value,
                     x_bins, y_bins, fig, axs[0, i], cmap, True, i == 0)
        axs[1, i].imshow(cropped_np_d2_img, extent=[d2_min[0], d2_max[0], d2_min[1], d2_max[1]])
        im = plot_heatmap(title, event, False, title_to_team_to_heatmap,
                     team_to_key_event_pos[False][0], max_bin_value,
                     x_bins, y_bins, fig, axs[1, i], cmap, False, i == 0)

    #fig.subplots_adjust(right=0.9)
    #cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, ax=axs)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(plots_path / (event + '.pdf'))
    with open(plots_path / (event + '.txt'), 'w') as f:
        f.write(f"num y points less than 1000: {title_to_num_y_points_less_than_1000}\n")
        f.write(f"num x points between negative 1200 and 0: {title_to_num_x_points_between_negative_1200_and_0}\n")
        f.write(f"num points: {title_to_num_points}\n")
