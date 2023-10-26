from typing import Optional, List

import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import generate_bins, plot_hist
from learn_bot.latent.analyze.compare_trajectories.run_trajectory_comparison import rollout_all_human_vs_learned_config
from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.load_data import LoadDataResult

config = rollout_all_human_vs_learned_config
load_data_result = LoadDataResult(config.predicted_load_data_options)
model = load_model_file(load_data_result)

time_to_hit_constraints: List[Optional[float]] = [None, 5., 3., 1., 0.8, 0.5, 0.25]
num_figs = len(time_to_hit_constraints)

fig_length = 6

fig = plt.figure(figsize=(fig_length*num_figs, fig_length), constrained_layout=True)
fig.suptitle("Human Velocity Metrics By Time To Hit Constraint")
axs = fig.subplots(1, num_figs, squeeze=False)

for i, time_to_hit_constraint in enumerate(time_to_hit_constraints):
    vels: List[pd.Series] = []
    for player_place_area_columns in specific_player_place_area_columns:
        constraint = model.cur_loaded_df[player_place_area_columns.alive] != 0
        if time_to_hit_constraint is not None:
            constraint = constraint & (
                ((model.cur_loaded_df[player_place_area_columns.seconds_until_next_hit_enemy] >= 0) &
                 (model.cur_loaded_df[player_place_area_columns.seconds_until_next_hit_enemy] < time_to_hit_constraint)) | \
                ((model.cur_loaded_df[player_place_area_columns.seconds_after_prior_hit_enemy] >= 0) &
                 (model.cur_loaded_df[player_place_area_columns.seconds_after_prior_hit_enemy] < time_to_hit_constraint))
            )
        filtered_df = model.cur_loaded_df[constraint]
        vels.append((filtered_df[player_place_area_columns.vel[0]] ** 2. +
                     filtered_df[player_place_area_columns.vel[1]] ** 2.) ** 0.5)

    axs[0, i].set_title(f'Constraint {time_to_hit_constraint}')
    bins: List
    bins = generate_bins(0, 300, 300 // 20)
    plot_hist(axs[0, i], pd.concat(vels), bins)
    axs[0, i].set_ylim(0., 1.)

plt.savefig(similarity_plots_path / 'velocity_metrics.png')
