import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from learn_bot.latent.place_area.column_names import PlayerPlaceAreaColumns
from learn_bot.latent.place_area.simulator import *


def compute_coverage_metrics(loaded_model: LoadedModel):
    num_ticks = 0
    num_alive_pats = 0
    x_pos_nps = []
    y_pos_nps = []
    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        print(f"Processing hdf5 {i + 1} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
        loaded_model.cur_hdf5_index = i
        loaded_model.load_cur_hdf5_as_pd(load_cur_dataset=False)

        num_ticks += len(loaded_model.cur_loaded_df)
        for player_columns in specific_player_place_area_columns:
            num_alive_pats += loaded_model.cur_loaded_df[player_columns.alive].sum()
            alive_df = loaded_model.cur_loaded_df[loaded_model.cur_loaded_df[player_columns.alive].astype('bool')]
            x_pos_nps.append(alive_df[player_columns.pos[0]].to_numpy())
            y_pos_nps.append(alive_df[player_columns.pos[1]].to_numpy())

    x_pos_np_concat = np.concatenate(x_pos_nps)
    y_pos_np_concat = np.concatenate(y_pos_nps)

    pos_heatmap, x_pos_bins, y_pos_bins = np.histogram2d(x_pos_np_concat, y_pos_np_concat, bins=250,
                                                         range=[[d2_min[0], d2_max[0]], [d2_min[1], d2_max[1]]])

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    fig.suptitle("Bot vs Human Metrics")
    ax = fig.subplots(1, 1)

    pos_heatmap = pos_heatmap.T

    grid_x, grid_y = np.meshgrid(x_pos_bins, y_pos_bins)
    hit_im = ax.pcolormesh(grid_x, grid_y, pos_heatmap)
    fig.colorbar(hit_im, ax=ax)
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
