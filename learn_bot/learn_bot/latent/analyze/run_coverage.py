import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from learn_bot.latent.place_area.simulation.simulator import *
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

plt.rc('font', family='Arial')

coverage_pickle_path = Path(__file__).parent / 'plots' / 'coverage.pickle'

per_player_pos_columns = \
    [player_place_area_columns.pos for player_place_area_columns in specific_player_place_area_columns]
pos_columns = flatten_list(per_player_pos_columns)
alive_columns = [player_place_area_columns.alive for player_place_area_columns in specific_player_place_area_columns]
first_tick_in_round_col = 'first tick in round'

def get_round_starts_np(loaded_model: LoadedModel) -> pd.DataFrame:
    id_df = loaded_model.get_cur_id_df().copy()
    round_ids_first_last_tick_id_df = \
        id_df.groupby(round_id_column, as_index=False).agg(
            first_tick=(tick_id_column, 'first'),
            num_ticks=(tick_id_column, 'count')
        )
    id_df = id_df.merge(round_ids_first_last_tick_id_df, on=round_id_column)
    return loaded_model.cur_dataset.X[id_df['first_tick'] == id_df[tick_id_column]]


def compute_coverage_metrics(loaded_model: LoadedModel, start_positions: bool):
    num_ticks = 0
    num_alive_pats = 0

    sum_pos_heatmap = None
    x_pos_bins = None
    y_pos_bins = None

    os.makedirs(coverage_pickle_path.parent, exist_ok=True)
    if True:
        with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
            for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
                #print(f"Processing hdf5 {i + 1} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
                loaded_model.cur_hdf5_index = i
                loaded_model.load_cur_dataset_only()

                if start_positions:
                    X_np = get_round_starts_np(loaded_model)
                else:
                    X_np = loaded_model.cur_dataset.X
                num_ticks += len(X_np)
                for player_index in range(len(specific_player_place_area_columns)):
                    alive_np = X_np[:, loaded_model.model.alive_columns[player_index]]
                    num_alive_pats += alive_np.astype(np.float64).sum()
                    X_alive_np = X_np[alive_np > 0.5]
                    x_pos = X_alive_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 0]]
                    y_pos = X_alive_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 1]]
                    if x_pos_bins is None:
                        sum_pos_heatmap, x_pos_bins, y_pos_bins = np.histogram2d(x_pos, y_pos, bins=125,
                                                                                 range=[[d2_min[0], d2_max[0]], [d2_min[1], d2_max[1]]])
                    else:
                        pos_heatmap, _, _ = np.histogram2d(x_pos, y_pos, bins=[x_pos_bins, y_pos_bins])
                        sum_pos_heatmap += pos_heatmap
                pbar.update(1)
        with open(coverage_pickle_path, "wb") as outfile:
            # "wb" argument opens the file in binary mode
            pickle.dump((sum_pos_heatmap, x_pos_bins, y_pos_bins), outfile)
    else:
        with open(coverage_pickle_path, "rb") as infile:
            (sum_pos_heatmap, x_pos_bins, y_pos_bins) = pickle.load(infile)

    fig = plt.figure(figsize=(3.25 * 0.49, 6/8 * 3.25 * 0.49), constrained_layout=True)
    if start_positions:
        fig.suptitle("Starting Positions", fontsize=8, x=0.35)
    else:
        fig.suptitle("All Positions", fontsize=8, x=0.35)
    ax = fig.subplots(1, 1)

    sum_pos_heatmap = sum_pos_heatmap.T

    grid_x, grid_y = np.meshgrid(x_pos_bins, y_pos_bins)

    heatmap_im = ax.pcolor(grid_x, grid_y, sum_pos_heatmap, norm=LogNorm(vmin=1, vmax=sum_pos_heatmap.max()),
                               cmap='turbo')
    heatmap_im.set_edgecolor('face')
    cbar = fig.colorbar(heatmap_im, ax=ax)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_ylabel('Player-Ticks', rotation=270, labelpad=7, fontsize=8)
    if start_positions:
        cbar.ax.set_yticks([1, 100, 10**3])
        yticklabels = cbar.ax.get_yticklabels()
        yticklabels[0] = "$\\mathdefault{1}$"
        #yticklabels[1] = "$\\mathdefault{10}$"
        cbar.ax.set_yticklabels(yticklabels)
    else:
        cbar.ax.set_yticks([1, 1000, 10**5])
        yticklabels = cbar.ax.get_yticklabels()
        yticklabels[0] = "$\\mathdefault{1}$"
        cbar.ax.set_yticklabels(yticklabels)
    cbar.ax.yaxis.set_tick_params(pad=0)
    cbar.ax.tick_params(axis="y", labelsize=8)

    # remove right/top spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    file_name = 'start_coverage.pdf' if start_positions else 'all_tick_coverage.pdf'

    print(f'saving {file_name}')
    plt.savefig(Path(__file__).parent / 'plots' / file_name)

    print(f"num ticks {num_ticks}, num alive player at ticks {num_alive_pats}")


if __name__ == "__main__":
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, use_test_data_only=False)
    print('computing coverage with all data')
    compute_coverage_metrics(loaded_model, False)
    print('computing coverage with just starts')
    compute_coverage_metrics(loaded_model, True)
