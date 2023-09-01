from dataclasses import field
from enum import Enum

import numpy as np
import pandas as pd
import torch

from learn_bot.latent.place_area.column_names import PlayerPlaceAreaColumns
from learn_bot.latent.place_area.simulator import *
# this is a open loop version of the simulator for computing metrics based on short time horizons

num_time_steps = 10


class PlayerMaskConfig(Enum):
    ALL = 0
    CT = 1
    T = 2
    LAST_ALIVE = 3


def compute_mask_elements_per_player(loaded_model: LoadedModel) -> int:
    return loaded_model.model.num_input_time_steps * loaded_model.model.num_dim

def build_player_mask(loaded_model: LoadedModel, config: PlayerMaskConfig,
                      round_lengths: RoundLengths) -> PlayerEnableMask:
    if config != PlayerMaskConfig.ALL:
        enabled_player_columns: List[List[bool]] = []
        if config == PlayerMaskConfig.CT:
            for _ in range(round_lengths.num_rounds):
                if round_lengths.ct_first:
                    enabled_player_columns.append([i in range(0, max_enemies) for i in range(0, 2*max_enemies)])
                else:
                    enabled_player_columns.append([i in range(max_enemies, 2*max_enemies)
                                                   for i in range(0, 2*max_enemies)])
        elif config == PlayerMaskConfig.T:
            for _ in range(round_lengths.num_rounds):
                if round_lengths.ct_first:
                    enabled_player_columns.append([i in range(max_enemies, 2*max_enemies)
                                                   for i in range(0, 2*max_enemies)])
                else:
                    enabled_player_columns.append([i in range(0, max_enemies) for i in range(0, 2*max_enemies)])
        elif config == PlayerMaskConfig.LAST_ALIVE:
            for _, last_alive in round_lengths.round_to_last_alive_index.items():
                enabled_player_columns.append([i == last_alive for i in range(0, 2 * max_enemies)])
        player_enable_mask = torch.tensor(enabled_player_columns)
        repeated_player_enable_mask = rearrange(repeat(
            rearrange(player_enable_mask, 'b p -> b p 1'), 'b p 1 -> b p r', r=compute_mask_elements_per_player(loaded_model)), 'b p r -> b (p r)')
        return repeated_player_enable_mask
    else:
        return None


def delta_pos_open_rollout(loaded_model: LoadedModel, round_lengths: RoundLengths,
                           player_enable_mask: PlayerEnableMask) -> PlayerEnableMask:
    rollout_tensor, similarity_tensor = \
        build_rollout_and_similarity_tensors(round_lengths, loaded_model.cur_dataset)
    pred_tensor = torch.zeros(rollout_tensor.shape[0], loaded_model.cur_dataset.Y.shape[1])
    loaded_model.model.eval()
    with torch.no_grad():
        num_steps = round_lengths.max_length_per_round - 1
        with tqdm(total=num_steps, disable=False) as pbar:
            for step_index in range(num_steps):
                if (step_index + 1) % num_time_steps != 0:
                    step(rollout_tensor, similarity_tensor, pred_tensor, loaded_model.model, round_lengths, step_index,
                         nav_data, player_enable_mask)
                pbar.update(1)
    # need to modify cur_loaded_df as rollout_df has constant length of all rounds for sim efficiency
    loaded_model.cur_loaded_df, loaded_model.cur_inference_df = \
        match_round_lengths(loaded_model.cur_loaded_df, rollout_tensor, pred_tensor, round_lengths,
                            loaded_model.column_transformers)
    return player_enable_mask


ground_truth_counter_column = 'ground truth counter'
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
def compare_predicted_rollout_indices(orig_df: pd.DataFrame, pred_df: pd.DataFrame, round_lengths: RoundLengths,
                                      player_enable_mask: PlayerEnableMask, loaded_model: LoadedModel) -> DisplacementErrors:
    round_lengths = get_round_lengths(loaded_model.cur_loaded_df)

    # get a counter for the ground truth group for each trajectory
    # first row will be ground truth, and rest will be relative to it
    # don't need this when comparing pred/orig since subtracting from each other will implicitly remove ground truth:
    # (pred delta + ground truth) - (orig delta + ground truth) = pred delta - orig delta
    # but need to determine which ground turth rows to filter out for ADE and what are last rows for FDE
    ground_truth_tick_indices = flatten_list([
        [idx for idx in round_subset_tick_indices if (idx - round_subset_tick_indices.start) % num_time_steps == 0]
        for _, round_subset_tick_indices in round_lengths.round_to_subset_tick_indices.items()
    ])
    ground_truth_tick_indicators = orig_df[tick_id_column] * 0
    ground_truth_tick_indicators.iloc[ground_truth_tick_indices] = 1
    ground_truth_counter = ground_truth_tick_indicators.cumsum()

    # use counter to compute first/last in each trajectory
    orig_df[ground_truth_counter_column] = ground_truth_counter
    orig_df[index_in_trajectory_column] = \
        orig_df.groupby(ground_truth_counter_column)[ground_truth_counter_column].transform('cumcount')
    # first if counter in trajectory is 0
    orig_df[is_ground_truth_column] = orig_df[index_in_trajectory_column] == 0

    result = DisplacementErrors()

    round_and_delta_df = pred_df.loc[:, [round_id_column]]

    # compute deltas
    for column_index, player_columns in enumerate(specific_player_place_area_columns):
        pred_vs_orig_delta_x = \
            pred_df[player_columns.pos[0]] - orig_df[player_columns.pos[0]]
        pred_vs_orig_delta_y = \
            pred_df[player_columns.pos[1]] - orig_df[player_columns.pos[1]]
        pred_vs_orig_delta_z = \
            pred_df[player_columns.pos[2]] - orig_df[player_columns.pos[2]]

        round_and_delta_df[pred_vs_orig_total_delta_column] = \
            (pred_vs_orig_delta_x ** 2. + pred_vs_orig_delta_y ** 2. + pred_vs_orig_delta_z ** 2.).pow(0.5)

        # last if counter in trajectory equals max index - need to recompute this for every player
        # as last will be different if die in middle of trajectory
        orig_df[index_in_trajectory_if_alive_column] = orig_df[index_in_trajectory_column]
        orig_df[index_in_trajectory_if_alive_column].where(orig_df[player_columns.alive].astype('bool'), -1)
        orig_df[last_pred_column] = \
            orig_df.groupby(ground_truth_counter_column)[index_in_trajectory_if_alive_column].transform('max')
        orig_df[is_last_pred_column] = orig_df[last_pred_column] == orig_df[index_in_trajectory_if_alive_column]

        all_pred_steps_df = round_and_delta_df[~orig_df[is_ground_truth_column] & orig_df[player_columns.alive]]
        final_pred_step_df = round_and_delta_df[orig_df[is_last_pred_column] & orig_df[player_columns.alive]]

        for round_index, round_id in enumerate(round_lengths.round_ids):
            if not player_enable_mask[round_index, column_index * compute_mask_elements_per_player(loaded_model)]:
                continue
            result.player_round_ades.append(all_pred_steps_df[all_pred_steps_df[round_id_column] == round_id].mean())
            result.player_round_fdes.append(final_pred_step_df[final_pred_step_df[round_id_column] == round_id].mean())

    return result


def run_analysis(loaded_model: LoadedModel):
    displacement_errors = DisplacementErrors()
    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        print(f"Processing hdf5 {i + 1} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
        loaded_model.cur_hdf5_index = i
        loaded_model.load_cur_hdf5_as_pd()

        # running rollout updates df, so keep original copy for analysis
        orig_loaded_df = loaded_model.cur_loaded_df.copy()
        round_lengths = get_round_lengths(loaded_model.cur_loaded_df)
        player_enable_mask = build_player_mask(loaded_model, PlayerMaskConfig.CT, round_lengths)
        delta_pos_open_rollout(loaded_model, round_lengths, player_enable_mask)

        hdf5_displacement_errors = compare_predicted_rollout_indices(orig_loaded_df, loaded_model.cur_loaded_df,
                                                                     round_lengths, player_enable_mask, loaded_model)
        displacement_errors.player_round_ades += hdf5_displacement_errors.player_round_ades
        displacement_errors.player_round_fdes += hdf5_displacement_errors.player_round_fdes

    ades = np.array(displacement_errors.player_round_ades)
    fdes = np.array(displacement_errors.player_round_fdes)

    print(f"ADE Mean: {ades.mean()}, ADE Std Dev: {ades.std()}, FDE Mean: {fdes.mean()}, FDE Std Dev: {fdes.std()}")


nav_data = None

perform_analysis = True

if __name__ == "__main__":
    nav_data = NavData(CUDA_DEVICE_STR)

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

    loaded_model = load_model_file(load_data_result, use_test_data_only=True)

    if perform_analysis:
        run_analysis(loaded_model)
    else:
        vis(loaded_model, delta_pos_open_rollout)
