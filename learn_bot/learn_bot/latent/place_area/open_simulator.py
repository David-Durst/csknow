import os
import sys
from dataclasses import field, dataclass
from enum import IntEnum
from math import floor
from pathlib import Path
from typing import List, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from tqdm import tqdm

from learn_bot.engagement_aim.column_names import tick_id_column
from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import plot_hist, generate_bins, \
    set_pd_print_options
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import NavData, data_ticks_per_second, \
    data_ticks_per_sim_tick
from learn_bot.latent.place_area.simulator import LoadedModel, RoundLengths, PlayerEnableMask, max_enemies, \
    build_rollout_and_similarity_tensors, save_inference_model_data_with_matching_round_lengths, step, \
    get_round_lengths, load_data_options, limit_to_every_nth_row
from learn_bot.latent.vis.vis import vis
from learn_bot.libs.io_transforms import CUDA_DEVICE_STR, flatten_list

# this is a open loop version of the simulator for computing metrics based on short time horizons

num_seconds_per_loop = 5
num_time_steps = data_ticks_per_second // data_ticks_per_sim_tick * num_seconds_per_loop


class PlayerMaskConfig(IntEnum):
    ALL = 0
    CT = 1
    T = 2
    LAST_ALIVE = 3
    CONSTANT_VELOCITY = 4
    NONE = 5
    NUM_MASK_CONFIGS = 6

    def __str__(self) -> str:
        if self == PlayerMaskConfig.ALL:
            return "All"
        if self == PlayerMaskConfig.CT:
            return "Offense Only"
        if self == PlayerMaskConfig.T:
            return "Defense Only"
        if self == PlayerMaskConfig.LAST_ALIVE:
            return "Last Alive"
        if self == PlayerMaskConfig.CONSTANT_VELOCITY:
            return "Constant Velocity"
        if self == PlayerMaskConfig.NONE:
            return "None"


# debugging flag where compute errors on players who are masked out (aka using ground truth)
include_ground_truth_players_in_afde = False


def compute_mask_elements_per_player(loaded_model: LoadedModel) -> int:
    return loaded_model.model.num_input_time_steps * loaded_model.model.num_dim


def build_player_mask(loaded_model: LoadedModel, config: PlayerMaskConfig,
                      round_lengths: RoundLengths) -> PlayerEnableMask:
    if config != PlayerMaskConfig.ALL and config != PlayerMaskConfig.CONSTANT_VELOCITY:
        enabled_player_columns: List[List[bool]] = []
        if config == PlayerMaskConfig.CT:
            for _ in range(round_lengths.num_rounds):
                if round_lengths.ct_first:
                    enabled_player_columns.append([i in range(0, max_enemies) for i in range(0, 2 * max_enemies)])
                else:
                    enabled_player_columns.append([i in range(max_enemies, 2 * max_enemies)
                                                   for i in range(0, 2 * max_enemies)])
        elif config == PlayerMaskConfig.T:
            for _ in range(round_lengths.num_rounds):
                if round_lengths.ct_first:
                    enabled_player_columns.append([i in range(max_enemies, 2 * max_enemies)
                                                   for i in range(0, 2 * max_enemies)])
                else:
                    enabled_player_columns.append([i in range(0, max_enemies) for i in range(0, 2 * max_enemies)])
        elif config == PlayerMaskConfig.LAST_ALIVE:
            for _, last_alive in round_lengths.round_to_last_alive_index.items():
                enabled_player_columns.append([i == last_alive for i in range(0, 2 * max_enemies)])
        elif config == PlayerMaskConfig.NONE:
            for _ in range(round_lengths.num_rounds):
                enabled_player_columns.append([False for _ in range(0, 2 * max_enemies)])
        player_enable_mask = torch.tensor(enabled_player_columns)
        repeated_player_enable_mask = rearrange(repeat(
            rearrange(player_enable_mask, 'b p -> b p 1'), 'b p 1 -> b p r',
            r=compute_mask_elements_per_player(loaded_model)), 'b p r -> b (p r)')
        return repeated_player_enable_mask
    else:
        return None


def build_constant_velocity_pred_tensor(loaded_model: LoadedModel, round_lengths: RoundLengths) -> torch.Tensor:
    pred_tensor_indices = []
    # for every tick in fixed length pred_tensor (sized to max length per round), get the index to read from actual Y
    for _, round_subset_tick_indices in round_lengths.round_to_subset_tick_indices.items():
        # if overrun end of round, just take last tick
        pred_tensor_indices += [min(round_subset_tick_indices[-1],
                                    # get first pred in num_time_steps trajectory
                                    round_subset_tick_indices[0] + step_index // num_time_steps * num_time_steps)
                                for step_index in range(round_lengths.max_length_per_round)]
    return loaded_model.cur_dataset.Y[pred_tensor_indices]


# round_lengths only accepts none so it can be called from vis, which handles many different sim functions
def delta_pos_open_rollout(loaded_model: LoadedModel, round_lengths: Optional[RoundLengths] = None,
                           player_enable_mask: PlayerEnableMask = None, constant_velocity: bool = False):
    rollout_tensor, similarity_tensor = \
        build_rollout_and_similarity_tensors(round_lengths, loaded_model.cur_dataset)
    if constant_velocity:
        pred_tensor = build_constant_velocity_pred_tensor(loaded_model, round_lengths)
    else:
        pred_tensor = torch.zeros(rollout_tensor.shape[0], loaded_model.cur_dataset.Y.shape[1])
    loaded_model.model.eval()
    with torch.no_grad():
        num_steps = round_lengths.max_length_per_round - 1
        with tqdm(total=num_steps, disable=False) as pbar:
            for step_index in range(num_steps):
                if (step_index + 1) % num_time_steps != 0:
                    step(rollout_tensor, similarity_tensor, pred_tensor, loaded_model.model, round_lengths, step_index,
                         nav_data, player_enable_mask, constant_velocity)
                pbar.update(1)
    save_inference_model_data_with_matching_round_lengths(loaded_model, rollout_tensor, pred_tensor, round_lengths)


def gen_vis_wrapper_delta_pos_open_rollout(player_mask_config: PlayerMaskConfig) -> Callable[[LoadedModel], None]:
    def result_func(loaded_model: LoadedModel):
        round_lengths = get_round_lengths(loaded_model.get_cur_id_df())
        player_enable_mask = build_player_mask(loaded_model, player_mask_config, round_lengths)
        delta_pos_open_rollout(loaded_model, round_lengths, player_enable_mask,
                               constant_velocity=player_mask_config == PlayerMaskConfig.CONSTANT_VELOCITY)
    return result_func


trajectory_counter_column = 'trajectory counter'
index_in_trajectory_column = 'index in trajectory'
index_in_trajectory_if_alive_column = 'index in trajectory if alive'
is_ground_truth_column = 'is ground truth'
last_pred_column = 'last pred'
is_last_pred_column = 'is last pred'
pred_vs_orig_total_delta_column = 'pred vs orig total delta'


@dataclass
class DisplacementErrors:
    player_round_ades: List[float] = field(default_factory=list)
    player_round_fdes: List[float] = field(default_factory=list)


# compute indices in open rollout that are actually predicted
def compare_predicted_rollout_indices(loaded_model: LoadedModel,
                                      player_enable_mask: PlayerEnableMask) -> DisplacementErrors:
    id_df = loaded_model.get_cur_id_df().copy()
    round_lengths = get_round_lengths(id_df)

    # get a counter for the ground truth group for each trajectory
    # first row will be ground truth, and rest will be relative to it
    # don't need this when comparing pred/orig since subtracting from each other will implicitly remove ground truth:
    # pred pos - orig pos = (pred delta + ground truth) - (orig delta + ground truth) = pred delta - orig delta
    # but need to determine which ground turth rows to filter out for ADE and what are last rows for FDE
    ground_truth_tick_indices = flatten_list([
        [idx for idx in round_subset_tick_indices if (idx - round_subset_tick_indices.start) % num_time_steps == 0]
        for _, round_subset_tick_indices in round_lengths.round_to_subset_tick_indices.items()
    ])
    ground_truth_tick_indicators = id_df[tick_id_column] * 0
    ground_truth_tick_indicators.iloc[ground_truth_tick_indices] = 1
    # new trajectory for each ground truth starting point
    trajectory_counter = ground_truth_tick_indicators.cumsum()

    # use counter to compute first/last in each trajectory
    id_df[trajectory_counter_column] = trajectory_counter
    id_df[index_in_trajectory_column] = \
        id_df.groupby(trajectory_counter_column)[trajectory_counter_column].transform('cumcount')
    # first if counter in trajectory is 0
    id_df[is_ground_truth_column] = id_df[index_in_trajectory_column] == 0

    result = DisplacementErrors()

    round_and_delta_df = id_df.loc[:, [round_id_column, index_in_trajectory_column, trajectory_counter_column]]

    # compute deltas
    for column_index in range(len(specific_player_place_area_columns)):
        pos_column_indices = loaded_model.model.nested_players_pos_columns_tensor[column_index, 0].tolist()
        alive_column_index = loaded_model.model.alive_columns[column_index]
        pred_vs_orig_delta_x = \
            loaded_model.cur_simulated_dataset.X[:, pos_column_indices[0]] - \
            loaded_model.cur_dataset.X[:, pos_column_indices[0]]
        pred_vs_orig_delta_y = \
            loaded_model.cur_simulated_dataset.X[:, pos_column_indices[1]] - \
            loaded_model.cur_dataset.X[:, pos_column_indices[1]]
        pred_vs_orig_delta_z = \
            loaded_model.cur_simulated_dataset.X[:, pos_column_indices[2]] - \
            loaded_model.cur_dataset.X[:, pos_column_indices[2]]

        round_and_delta_df[pred_vs_orig_total_delta_column] = \
            (pred_vs_orig_delta_x ** 2. + pred_vs_orig_delta_y ** 2. + pred_vs_orig_delta_z ** 2.) ** 0.5

        # last if counter in trajectory equals max index - need to recompute this for every player
        # as last will be different if die in middle of trajectory
        id_df[index_in_trajectory_if_alive_column] = id_df[index_in_trajectory_column]
        player_alive_series = pd.Series(loaded_model.cur_dataset.X[:, alive_column_index] == 1)
        id_df[index_in_trajectory_if_alive_column].where(player_alive_series, -1, inplace=True)
        id_df[last_pred_column] = \
            id_df.groupby(trajectory_counter_column)[index_in_trajectory_if_alive_column].transform('max')
        id_df[is_last_pred_column] = id_df[last_pred_column] == id_df[index_in_trajectory_if_alive_column]

        all_pred_steps_df = round_and_delta_df[~id_df[is_ground_truth_column] & player_alive_series]
        # need to remove ground truth, otherwise include FDE for 1 tick trajecotries where first tick is ground truth
        # and die before can make any predictions (no nothing worth evaluating, just get incorrect 0 FDE)
        final_pred_step_df = round_and_delta_df[id_df[is_last_pred_column] & ~id_df[is_ground_truth_column] &
                                                player_alive_series]

        ade_traj_ids = all_pred_steps_df[trajectory_counter_column].unique()
        fde_traj_ids = final_pred_step_df[trajectory_counter_column].unique()
        if len(ade_traj_ids) != len(fde_traj_ids):
            print('missing traj')

        # get ade by averaging within trajectories only. This will enable looking at per-trajectory ADE distribution.
        # Aggregating across trajectories prevent that type of distributional analysis.
        # Same round_id across all ticks in a trajectory, so just take first
        player_round_ades = all_pred_steps_df.groupby(trajectory_counter_column).agg(
            {round_id_column: 'first', pred_vs_orig_total_delta_column: 'mean'})
        player_round_fdes = final_pred_step_df.groupby(trajectory_counter_column).agg(
            {round_id_column: 'first', pred_vs_orig_total_delta_column: 'mean'})
        player_valid_round_ids = []
        for round_index, round_id in enumerate(round_lengths.round_ids):
            if include_ground_truth_players_in_afde or player_enable_mask is None or \
                    player_enable_mask[round_index, column_index * compute_mask_elements_per_player(loaded_model)]:
                player_valid_round_ids.append(round_id)
        result.player_round_ades += list(player_round_ades[player_round_ades[round_id_column]
                                         .isin(player_valid_round_ids)][pred_vs_orig_total_delta_column])
        result.player_round_fdes += list(player_round_fdes[player_round_fdes[round_id_column]
                                         .isin(player_valid_round_ids)][pred_vs_orig_total_delta_column])
        tmp_ades = list(player_round_ades[player_round_ades[round_id_column]
                                         .isin(player_valid_round_ids)][pred_vs_orig_total_delta_column])
        tmp_fdes = list(player_round_fdes[player_round_fdes[round_id_column]
                                         .isin(player_valid_round_ids)][pred_vs_orig_total_delta_column])
        if len(tmp_ades) != len(tmp_fdes):
            print('length mismatch')
        # 2500 possible (not bug) - can go 250 per second for 5 seconds is 1250, mul by 2 because pred and orig can go
        # in opposite directions
        # 200+ more is possible if fall off ledge, but haven't seen that on going in opposite direction
        #if max(tmp_fdes) > 2500.:
        #    bad_round_and_trajectory = player_round_fdes[player_round_fdes[pred_vs_orig_total_delta_column] > 2000]
        #    bad_trajectory_id = bad_round_and_trajectory.index[0]
        #    bad_orig_positions = orig_df[orig_df[trajectory_counter_column] == bad_trajectory_id] \
        #                             .loc[:, [tick_id_column, player_columns.alive, player_columns.pos[0], player_columns.pos[1], player_columns.pos[2]]]
        #    bad_pred_positions = pred_df[orig_df[trajectory_counter_column] == bad_trajectory_id] \
        #                             .loc[:, [tick_id_column, player_columns.alive, player_columns.pos[0], player_columns.pos[1], player_columns.pos[2]]]
        #    bad_delta = round_and_delta_df[orig_df[trajectory_counter_column] == bad_trajectory_id] \
        #                    .loc[:, [pred_vs_orig_total_delta_column]]
        #    print('invalid max')

    return result


num_bins = 40
num_iterations = 6


def plot_ade_fde(player_mask_configs: List[PlayerMaskConfig], displacement_errors: list[pd.Series], axs, is_ade: bool):
    min_value = int(floor(min([de.min() for de in displacement_errors])))
    max_value = int(floor(max([de.max() for de in displacement_errors])))
    bin_width = (max_value - min_value) // num_bins
    bins = generate_bins(min_value, max_value, bin_width)
    for player_mask_config, de, ax in zip(player_mask_configs, displacement_errors, axs):
        plot_hist(ax, de, bins)
        ax.text((min_value + max_value) / 2., 0.4, de.describe().to_string(), family='monospace')
        ax.set_ylim(0., 1.)
        ax.set_xlim(min(0., min_value), max_value)
        ax.set_title(str(player_mask_config) + (" minADE" if is_ade else " minFDE"))


def run_analysis_per_mask(loaded_model: LoadedModel, player_mask_config: PlayerMaskConfig) -> \
        Tuple[pd.Series, pd.Series]:
    displacement_errors = DisplacementErrors()
    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        if i > 1:
            break

        print(f"Processing hdf5 {i} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
        per_iteration_displacement_errors: List[DisplacementErrors] = []

        loaded_model.cur_hdf5_index = i
        loaded_model.load_cur_dataset_only()

        for iteration in range(num_iterations):
            print(f"iteration {iteration} / {num_iterations}")

            round_lengths = get_round_lengths(loaded_model.get_cur_id_df())
            player_enable_mask = build_player_mask(loaded_model, player_mask_config, round_lengths)
            delta_pos_open_rollout(loaded_model, round_lengths, player_enable_mask,
                                   constant_velocity=player_mask_config == PlayerMaskConfig.CONSTANT_VELOCITY)

            hdf5_displacement_errors = compare_predicted_rollout_indices(loaded_model, player_enable_mask)
            per_iteration_displacement_errors.append(hdf5_displacement_errors)

        per_iteration_ade: List[List[float]] = [de.player_round_ades for de in per_iteration_displacement_errors]
        displacement_errors.player_round_ades += list(np.min(np.array(per_iteration_ade), axis=0))

        per_iteration_fde: List[List[float]] = [de.player_round_fdes for de in per_iteration_displacement_errors]
        displacement_errors.player_round_fdes += list(np.min(np.array(per_iteration_fde), axis=0))

    ades = pd.Series(displacement_errors.player_round_ades)
    fdes = pd.Series(displacement_errors.player_round_fdes)

    return ades, fdes


simulation_plots_path = Path(__file__).parent / 'simulation_plots'
fig_length = 8
num_metrics = 2


def run_analysis(loaded_model: LoadedModel):
    os.makedirs(simulation_plots_path, exist_ok=True)

    fig = plt.figure(figsize=(fig_length * PlayerMaskConfig.NUM_MASK_CONFIGS, fig_length * num_metrics),
                     constrained_layout=True)
    fig.suptitle("Model-Level Metrics")
    axs = fig.subplots(num_metrics, PlayerMaskConfig.NUM_MASK_CONFIGS, squeeze=False)

    mask_result_strs = []
    mask_result_latex_strs = ["Simulation Type & minADE Mean & minADE Std Dev & minFDE Mean & minFDE Std Dev \\\\",
                              "\\hline"]
    player_mask_configs = [PlayerMaskConfig.ALL,
                           #PlayerMaskConfig.CT, PlayerMaskConfig.T,
                           #PlayerMaskConfig.LAST_ALIVE,
                           PlayerMaskConfig.CONSTANT_VELOCITY]
    ades_per_mask_config: List[pd.Series] = []
    fdes_per_mask_config: List[pd.Series] = []
    for i, player_mask_config in enumerate(player_mask_configs):
        print(f"Config {player_mask_config}")
        ades, fdes = run_analysis_per_mask(loaded_model, player_mask_config)
        ades_per_mask_config.append(ades)
        fdes_per_mask_config.append(fdes)
        mask_result_str = f"{str(player_mask_config)} minADE Mean: {ades.mean():.2f}, " \
                          f"minADE Std Dev: {ades.std():.2f}, " \
                          f"minFDE Mean: {fdes.mean():.2f}, minFDE Std Dev: {fdes.std():.2f}"
        mask_result_strs.append(mask_result_str)
        mask_result_latex_str = f"{str(player_mask_config)} & {ades.mean():.2f} & {ades.std():.2f} & " \
                                f"{fdes.mean():.2f} & {fdes.std():.2f} \\\\"
        mask_result_latex_strs.append(mask_result_latex_str)
        print(mask_result_str)

    plot_ade_fde(player_mask_configs, ades_per_mask_config, axs[0], True)
    plot_ade_fde(player_mask_configs, fdes_per_mask_config, axs[1], False)
    print('\n'.join(mask_result_strs))
    print('\n'.join(mask_result_latex_strs))

    plt.savefig(simulation_plots_path / 'ade_fde_by_mask.png')


nav_data = None
perform_analysis = True
vis_player_mask_config = PlayerMaskConfig.ALL

if __name__ == "__main__":
    if len(sys.argv) > 2:
        perform_analysis = sys.argv[2] == "True"

    nav_data = NavData(CUDA_DEVICE_STR)

    load_data_options.custom_limit_fn = limit_to_every_nth_row
    load_data_result = LoadDataResult(load_data_options)
    # if manual_data:
    #    all_data_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=[i for i in range(20000)])
    #    #all_data_df = all_data_df[all_data_df['test name'] == b'LearnedGooseToCatScript']
    # elif rollout_data:
    #    all_data_df = load_hdf5_to_pd(rollout_latent_team_hdf5_data_path)
    # else:
    #    all_data_df = load_hdf5_to_pd(human_latent_team_hdf5_data_path, rows_to_get=[i for i in range(20000)])
    # all_data_df = all_data_df.copy()

    # load_result = load_model_file_for_rollout(all_data_df, "delta_pos_checkpoint.pt")

    # named outer as uusing loaded_model in lots of parameters to functions as well
    outer_loaded_model = load_model_file(load_data_result, use_test_data_only=True)

    set_pd_print_options()

    if perform_analysis:
        run_analysis(outer_loaded_model)
    else:
        vis(outer_loaded_model, gen_vis_wrapper_delta_pos_open_rollout(vis_player_mask_config), " Open Loop Simulator",
            use_sim_dataset=True)
