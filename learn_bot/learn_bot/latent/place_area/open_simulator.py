import numpy as np
import pandas as pd
import torch

from learn_bot.latent.place_area.column_names import PlayerPlaceAreaColumns
from learn_bot.latent.place_area.simulator import *
# this is a open loop version of the simulator for computing metrics based on short time horizons

num_time_steps = 10


def delta_pos_open_rollout(loaded_model: LoadedModel):
    round_lengths = get_round_lengths(loaded_model.cur_loaded_df)
    rollout_tensor, similarity_tensor = \
        build_rollout_and_similarity_tensors(round_lengths, loaded_model.cur_dataset)
    pred_tensor = torch.zeros(rollout_tensor.shape[0], loaded_model.cur_dataset.Y.shape[1])
    loaded_model.model.eval()
    with torch.no_grad():
        num_steps = round_lengths.max_length_per_round - 1
        with tqdm(total=num_steps, disable=False) as pbar:
            for step_index in range(num_steps):
                if (step_index + 1) % num_time_steps != 0:
                    step(rollout_tensor, similarity_tensor, pred_tensor, loaded_model.model, round_lengths, step_index, nav_data)
                pbar.update(1)
    # need to modify cur_loaded_df as rollout_df has constant length of all rounds for sim efficiency
    loaded_model.cur_loaded_df, loaded_model.cur_inference_df = \
        match_round_lengths(loaded_model.cur_loaded_df, rollout_tensor, pred_tensor, round_lengths,
                            loaded_model.column_transformers)


ground_truth_counter_column = 'ground truth counter'
index_in_trajectory_column = 'index in trajectory'
index_in_trajectory_if_alive_column = 'index in trajectory if alive'
is_ground_truth_column = 'is ground truth'
last_pred_column = 'last pred'
is_last_pred_column = 'is last pred'


@dataclass
class DisplacementErrors:
    ade: float
    fde: float
    num_elements_in_sum: int


# compute indices in open rollout that are actually predicted
def compare_predicted_rollout_indices(orig_df: pd.DataFrame, pred_df: pd.DataFrame) -> DisplacementErrors:
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

    result = DisplacementErrors(0., 0., 0)

    # compute deltas
    for player_columns in specific_player_place_area_columns:
        # orig_df[player_columns.pos[1]] - orig_df_grouped[player_columns.pos[1]].transform('first')
        pred_vs_orig_delta_x = \
            pred_df[player_columns.pos[0]] - orig_df[player_columns.pos[0]]
        pred_vs_orig_delta_y = \
            pred_df[player_columns.pos[1]] - orig_df[player_columns.pos[1]]
        pred_vs_orig_delta_z = \
            pred_df[player_columns.pos[2]] - orig_df[player_columns.pos[2]]

        pred_vs_orig_total_delta = (pred_vs_orig_delta_x ** 2. + pred_vs_orig_delta_y ** 2. +
                                    pred_vs_orig_delta_z ** 2.).pow(0.5)

        # last if counter in trajectory equals max index - need to recompute this for every player
        # as last will be different if die in middle of trajectory
        orig_df[index_in_trajectory_if_alive_column] = orig_df[index_in_trajectory_column]
        orig_df[index_in_trajectory_if_alive_column].where(orig_df[player_columns.alive], -1)
        orig_df[last_pred_column] = \
            orig_df.groupby(ground_truth_counter_column)[index_in_trajectory_if_alive_column].transform('max')
        orig_df[is_last_pred_column] = orig_df[last_pred_column] == orig_df[index_in_trajectory_if_alive_column]

        result.ade += pred_vs_orig_total_delta[~orig_df[is_ground_truth_column] & orig_df[player_columns.alive]].mean()
        result.fde += pred_vs_orig_total_delta[orig_df[is_last_pred_column] & orig_df[player_columns.alive]].mean()
        # doing averaging for variable length sequences here (different number of ticks where each player column alive)
        # do fixed length averaging across players in one place at end
        result.num_elements_in_sum += 1

    return result


def run_analysis(loaded_model: LoadedModel):
    displacement_errors = DisplacementErrors(0., 0., 0)
    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        print(f"Processing hdf5 {i + 1} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
        loaded_model.cur_hdf5_index = i
        loaded_model.load_cur_hdf5_as_pd()

        # running rollout updates df, so keep original copy for analysis
        orig_loaded_df = loaded_model.cur_loaded_df.copy()
        delta_pos_open_rollout(loaded_model)

        hdf5_displacement_errors = compare_predicted_rollout_indices(orig_loaded_df, loaded_model.cur_loaded_df)
        displacement_errors.ade += hdf5_displacement_errors.ade
        displacement_errors.fde += hdf5_displacement_errors.fde
        displacement_errors.num_elements_in_sum += hdf5_displacement_errors.num_elements_in_sum

    displacement_errors.ade /= displacement_errors.num_elements_in_sum
    displacement_errors.fde /= displacement_errors.num_elements_in_sum

    print(f"ADE: {displacement_errors.ade}, FDE: {displacement_errors.fde}")


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
