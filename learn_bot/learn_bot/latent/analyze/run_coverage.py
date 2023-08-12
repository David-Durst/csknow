import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

from learn_bot.latent.place_area.column_names import PlayerPlaceAreaColumns
from learn_bot.latent.place_area.simulator import *


def compute_coverage_metrics(loaded_model: LoadedModel):
    num_ticks = 0
    num_alive_pats = 0

    sum_pos_heatmap = None
    x_pos_bins = None
    y_pos_bins = None

    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        print(f"Processing hdf5 {i + 1} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
        loaded_model.cur_hdf5_index = i
        loaded_model.load_cur_hdf5_as_pd(load_cur_dataset=False)

        num_ticks += len(loaded_model.cur_loaded_df)
        for player_columns in specific_player_place_area_columns:
            num_alive_pats += loaded_model.cur_loaded_df[player_columns.alive].sum()
            alive_df = loaded_model.cur_loaded_df[loaded_model.cur_loaded_df[player_columns.alive].astype('bool')]
            x_pos = alive_df[player_columns.pos[0]].to_numpy()
            y_pos = alive_df[player_columns.pos[1]].to_numpy()
            if x_pos_bins is None:
                sum_pos_heatmap, x_pos_bins, y_pos_bins = np.histogram2d(x_pos, y_pos, bins=125,
                                                                         range=[[d2_min[0], d2_max[0]], [d2_min[1], d2_max[1]]])
            else:
                pos_heatmap, _, _ = np.histogram2d(x_pos, y_pos, bins=[x_pos_bins, y_pos_bins])
                sum_pos_heatmap += pos_heatmap

        if i == 4:
            break


    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    fig.suptitle("Bot vs Human Metrics")
    ax = fig.subplots(1, 1)

    sum_pos_heatmap = sum_pos_heatmap.T

    grid_x, grid_y = np.meshgrid(x_pos_bins, y_pos_bins)
    colors_not_occupied = plt.cm.magma(np.linspace(0, 0.25, 256))
    colors_occupied = plt.cm.viridis(np.linspace(0.25, 1, 256))
    all_colors = np.vstack((colors_not_occupied, colors_occupied))
    not_to_occupied_map = LinearSegmentedColormap.from_list(
        'not_to_occupied_map', all_colors)

    heatmap_im = ax.pcolormesh(grid_x, grid_y, sum_pos_heatmap,
                           norm=TwoSlopeNorm(vmin=0, vcenter=2, vmax=sum_pos_heatmap.max()),
                           cmap=not_to_occupied_map)
    fig.colorbar(heatmap_im, ax=ax)
    ax.set_title(f"Hit Aim With Recoil Distribution")
    ax.set_xlabel("Normalized Yaw Distance (1 = AABB width)")
    ax.set_ylabel("Normalized Pitch Distance (1 = AABB height)")

    plt.savefig(Path(__file__).parent / 'plots' / 'coverage.png')

    print(f"num ticks {num_ticks}, num alive player at ticks {num_alive_pats}")


if __name__ == "__main__":
    load_data_result = LoadDataResult(load_data_options)
    #if manual_data:
    #    all_data_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=[i for i in range(20000)])
    #    #all_data_df = all_data_df[all_data_df['test name'] == b'LearnedGooseToCatScript']
    #elif rollout_data:
    #    all_data_df = load_hdf5_to_pd(rollout_latent_team_hdf5_data_path)
    #else:
    #    all_data_df = load_hdf5_to_pd(human_latent_team_hdf5_data_path, rows_to_get=[i for i in range(20000)])
    #all_data_df = all_data_df.copy()

    #load_result = load_model_file_for_rollout(all_data_df, "delta_pos_checkpoint.pt")

    loaded_model = load_model_file(load_data_result, use_test_data_only=False)
    compute_coverage_metrics(loaded_model)
