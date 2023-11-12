import os
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from learn_bot.latent.place_area.simulation.simulator import *

coverage_pickle_path = Path(__file__).parent / 'plots' / 'coverage.pickle'

def compute_coverage_metrics(loaded_model: LoadedModel):
    num_ticks = 0
    num_alive_pats = 0

    sum_pos_heatmap = None
    x_pos_bins = None
    y_pos_bins = None

    os.makedirs(coverage_pickle_path.parent, exist_ok=True)
    if True:
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
        with open(coverage_pickle_path, "wb") as outfile:
            # "wb" argument opens the file in binary mode
            pickle.dump((sum_pos_heatmap, x_pos_bins, y_pos_bins), outfile)
    else:
        with open(coverage_pickle_path, "rb") as infile:
            (sum_pos_heatmap, x_pos_bins, y_pos_bins) = pickle.load(infile)



    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    fig.suptitle("Data Set Coverage Of de_dust2 Map", fontsize=16)
    ax = fig.subplots(1, 1)

    sum_pos_heatmap = sum_pos_heatmap.T

    grid_x, grid_y = np.meshgrid(x_pos_bins, y_pos_bins)

    heatmap_im = ax.pcolormesh(grid_x, grid_y, sum_pos_heatmap, norm=LogNorm(vmin=1, vmax=sum_pos_heatmap.max()),
                               cmap='viridis')
    cbar = fig.colorbar(heatmap_im, ax=ax)
    cbar.ax.set_ylabel('Number of Per-Player Data Points', rotation=270, labelpad=15, fontsize=14)


    plt.savefig(Path(__file__).parent / 'plots' / 'coverage.png')

    print(f"num ticks {num_ticks}, num alive player at ticks {num_alive_pats}")


if __name__ == "__main__":
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, use_test_data_only=False)
    compute_coverage_metrics(loaded_model)
