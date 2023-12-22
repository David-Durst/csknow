import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import title_to_team_to_pos_dict
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.transformer_nested_hidden_latent_model import d2_min, d2_max

fig_length = 10

title_to_num_y_points_less_than_1000: Dict[str, int] = {}
title_to_num_x_points_between_negative_1200_and_0: Dict[str, int] = {}

def create_heatmap(title: str, ct_team: bool, title_to_team_to_heatmap: Dict[str, Dict],
                   x_pos: List[float], y_pos: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    heatmap, x_bins, y_bins = np.histogram2d(x_pos, y_pos, bins=50,
                                             range=[[d2_min[0], d2_max[0]], [d2_min[1], d2_max[1]]])
    if title not in title_to_num_y_points_less_than_1000:
        title_to_num_y_points_less_than_1000[title] = 0
        title_to_num_x_points_between_negative_1200_and_0[title] = 0
    title_to_num_y_points_less_than_1000[title] += len([y for y in y_pos if y < 1000])
    title_to_num_x_points_between_negative_1200_and_0[title] += len([x for x in x_pos if x > -1200 and x < 0])
    heatmap = heatmap.T
    if title not in title_to_team_to_heatmap:
        title_to_team_to_heatmap[title] = {}
    title_to_team_to_heatmap[title][ct_team] = heatmap
    return x_bins, y_bins


def plot_heatmap(title: str, event: str, ct_team: bool, title_to_team_to_heatmap: Dict[str, Dict],
                 x_pos: List[float], max_bin_value: int,
                 x_bins: np.ndarray, y_bins: np.ndarray, fig: plt.Figure, ax: plt.Axes, cmap):
    ax.set_title(f"{title} {'CT' if ct_team else 'T'} {event} ({len(x_pos)} points)")
    grid_x, grid_y = np.meshgrid(x_bins, y_bins)
    heatmap_im = ax.pcolormesh(grid_x, grid_y, title_to_team_to_heatmap[title][ct_team], vmin=1,
                               vmax=max_bin_value, cmap=cmap)
    cbar = fig.colorbar(heatmap_im, ax=ax)
    cbar.ax.set_ylabel(f'Number of {event}', rotation=270, labelpad=15, fontsize=14)


def plot_key_event_heatmaps(title_to_team_to_key_event_pos: title_to_team_to_pos_dict,
                            trajectory_filter_options: TrajectoryFilterOptions, plots_path: Path):
    if not trajectory_filter_options.filtering_key_events():
        return

    num_titles = len(title_to_team_to_key_event_pos.keys())
    # mul cols by 2 as need ct and t plots
    fig = plt.figure(figsize=(fig_length * 2, fig_length * num_titles), constrained_layout=True)
    axs = fig.subplots(num_titles, 2, squeeze=False)

    cmap = plt.get_cmap('tab20b').copy()
    cmap.set_under('gray')

    max_points = 0
    for _, team_to_key_event_pos in title_to_team_to_key_event_pos.items():
        max_points = max(max_points, len(team_to_key_event_pos[True][0]))
        max_points = max(max_points, len(team_to_key_event_pos[False][0]))

    event = ""
    if trajectory_filter_options.only_kill:
        event = "Kills"
    elif trajectory_filter_options.only_killed:
        event = "Deaths"
    elif trajectory_filter_options.only_killed_or_end:
        event = "DeathsAndEnds"
    elif trajectory_filter_options.only_shots:
        event = "Shots"

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
        plot_heatmap(title, event, True, title_to_team_to_heatmap,
                     team_to_key_event_pos[True][0], max_bin_value,
                     x_bins, y_bins, fig, axs[i, 0], cmap)
        plot_heatmap(title, event, False, title_to_team_to_heatmap,
                     team_to_key_event_pos[False][0], max_bin_value,
                     x_bins, y_bins, fig, axs[i, 1], cmap)
    plt.savefig(plots_path / (event + '.png'))
    print(title_to_num_y_points_less_than_1000)
    print(title_to_num_x_points_between_negative_1200_and_0)
