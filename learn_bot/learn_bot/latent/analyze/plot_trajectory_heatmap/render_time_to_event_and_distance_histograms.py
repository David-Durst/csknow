from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import crosshair_distance_to_enemy_col, \
    world_distance_to_enemy_col
from learn_bot.latent.analyze.plot_trajectory_heatmap.title_rename_dict import title_rename_dict


@dataclass
class EventDistanceHistogramData:
    avg_heatmap: np.ndarray
    x_bins: np.ndarray
    y_bins: np.ndarray


def bin_one_title_time_to_event_and_distance_histograms(title_to_values: Dict[str, List[Dict[str, float]]],
                                                        time_to_event_col: str, title: str) -> EventDistanceHistogramData:
    df = pd.DataFrame.from_records(title_to_values[title])
    # every 3 degreees, 10 bins for crosshair distance
    crosshair_bins = [i*3. for i in range(11)]
    # every 100 units, 20 bins for world distance
    world_bins = [i*100. for i in range(21)]

    # get avg time to event per bin heatmap
    sums_heatmap, x_bins, y_bins = np.histogram2d(df[crosshair_distance_to_enemy_col], df[world_distance_to_enemy_col],
                                                  weights=df[time_to_event_col], bins=(crosshair_bins, world_bins))
    counts_heatmap, _, _ = np.histogram2d(df[crosshair_distance_to_enemy_col], df[world_distance_to_enemy_col],
                                          bins=(crosshair_bins, world_bins))
    sums_heatmap = sums_heatmap.T
    counts_heatmap = counts_heatmap.T
    sums_heatmap[counts_heatmap == 0] = -1
    counts_heatmap[counts_heatmap == 0] = 1
    avg_heatmap = sums_heatmap / counts_heatmap

    return EventDistanceHistogramData(avg_heatmap, x_bins, y_bins)


def plot_one_title_time_to_event_and_distance_histograms(event_distance_histogram_data: EventDistanceHistogramData,
                                                         max_bin_value: float, ax: plt.Axes, cmap, title: str):

    # plot heatmap
    grid_x, grid_y = np.meshgrid(event_distance_histogram_data.x_bins, event_distance_histogram_data.y_bins)
    heatmap_im = ax.pcolor(grid_x, grid_y, event_distance_histogram_data.avg_heatmap, vmin=0, vmax=max_bin_value,
                           cmap=cmap)
    heatmap_im.set_edgecolor('face')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_title(title_rename_dict[title], fontsize=8)
    return heatmap_im


def compute_time_to_event_and_distance_histograms(title_to_values: Dict[str, List[Dict[str, float]]], metric_title: str,
                                                  time_to_event_col, plot_path: Path):
    num_titles = len(title_to_values.keys())
    if num_titles != 4:
        return
    fig = plt.figure(figsize=(7, 7))#, constrained_layout=True)
    axs = fig.subplots(2, 2, squeeze=False)

    cmap = plt.get_cmap('tab20b').copy()
    cmap.set_under((1, 1, 1, 0))

    title_to_data: Dict[str, EventDistanceHistogramData] = {}
    max_bin_value = 0
    for title, values in title_to_values.items():
        title_to_data[title] = bin_one_title_time_to_event_and_distance_histograms(title_to_values, time_to_event_col,
                                                                                   title)
        max_bin_value = max(max_bin_value, title_to_data[title].avg_heatmap.max())
    for i, (title, data) in enumerate(title_to_data.items()):
        r = i % 2
        c = i // 2
        im = plot_one_title_time_to_event_and_distance_histograms(data, max_bin_value, axs[r, c], cmap, title)
    fig.suptitle(metric_title, fontsize=8)
    fig.supxlabel('World Distance', fontsize=8)
    fig.supylabel('Crosshair Distance', fontsize=8)
    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.95, hspace=0.35)
    cbar = fig.colorbar(im, ax=axs)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Seconds')
    fig.savefig(plot_path / (metric_title.lower().replace(' ', '_') + '.pdf'))


