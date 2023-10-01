from pathlib import Path

import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from tqdm import tqdm

from learn_bot.latent.analyze.comparison_column_names import small_human_good_rounds, \
    all_human_28_second_filter_good_rounds, all_human_vs_small_human_similarity_hdf5_data_path, \
    all_human_vs_human_28_similarity_hdf5_data_path
from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import round_id_column, tick_id_column
from learn_bot.latent.load_model import load_model_file, LoadedModel
from learn_bot.latent.order.column_names import num_radial_ticks
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import delta_pos_grid_num_cells_per_xy_dim, \
    delta_pos_grid_cell_dim, \
    delta_pos_grid_num_xy_cells_per_z_change, compute_new_pos, NavData, data_ticks_per_sim_tick
from learn_bot.latent.place_area.load_data import human_latent_team_hdf5_data_path, manual_latent_team_hdf5_data_path, \
    rollout_latent_team_hdf5_data_path, LoadDataResult, LoadDataOptions, SimilarityFn
from learn_bot.latent.train_paths import train_test_split_file_name
from learn_bot.latent.transformer_nested_hidden_latent_model import *
from learn_bot.latent.vis.vis import vis
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd, load_hdf5_extra_to_list
from learn_bot.libs.io_transforms import get_untransformed_outputs, CPU_DEVICE_STR, get_label_outputs, \
    get_transformed_outputs


@dataclass
class RoundLengths:
    num_rounds: int
    max_length_per_round: int
    round_ids: List[int]
    round_to_tick_ids: Dict[int, range]
    # tick indices for subset (ie. if just test data or entire data, different indices)
    round_to_subset_tick_indices: Dict[int, range]
    round_to_length: Dict[int, int]
    round_id_to_list_id: Dict[int, int]
    ct_first: bool
    round_to_last_alive_index: Dict[int, int]


def limit_to_every_nth_row(df: pd.DataFrame):
    grouped_df = df.groupby([round_id_column]).agg({tick_id_column: 'min'})
    condition = df[tick_id_column] != df[tick_id_column]
    for round_id, round_row in grouped_df.iterrows():
        condition = condition | ((df[round_id_column] == round_id) &
                                 ((df[tick_id_column] - round_row[tick_id_column]) % data_ticks_per_sim_tick == 0))
    return condition


def get_round_lengths(df: pd.DataFrame, compute_last_player_alive: bool = True) -> RoundLengths:
    grouped_df = df.groupby([round_id_column]).agg({tick_id_column: ['count', 'min', 'max'],
                                                    'index': ['count', 'min', 'max']})
    result = RoundLengths(len(grouped_df), max(grouped_df[tick_id_column]['count']), [], {}, {}, {}, {}, False, {})
    list_id = 0
    for round_id, round_row in grouped_df[tick_id_column].iterrows():
        result.round_ids.append(round_id)
        result.round_to_tick_ids[round_id] = range(round_row['min'], round_row['max'] + 1, data_ticks_per_sim_tick)
        result.round_to_length[round_id] = len(result.round_to_tick_ids[round_id])
        result.round_id_to_list_id[round_id] = list_id

        last_row = df.loc[grouped_df['index'].loc[round_id, 'max']]
        if compute_last_player_alive:
            for column_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
                if last_row[player_place_area_columns.alive]:
                    result.round_to_last_alive_index[round_id] = column_index
                    break
            if round_id not in result.round_to_last_alive_index:
                print(f'no one alive at end of {round_id}')

        list_id += 1
    result.ct_first = team_strs[0] in specific_player_place_area_columns[0].player_id

    for round_id, round_row in grouped_df['index'].iterrows():
        result.round_to_subset_tick_indices[round_id] = range(round_row['min'], round_row['max']+1)

    return result


# src tensor is variable length per round, rollout tensor is fixed length for efficiency
# fillout rollout tensor for as much as possible for each round so have non-sim input features (like visibility)
def build_rollout_and_similarity_tensors(round_lengths: RoundLengths, dataset: LatentDataset) -> \
        Tuple[torch.Tensor,torch.Tensor]:
    rollout_tensor = torch.zeros([round_lengths.num_rounds * round_lengths.max_length_per_round, dataset.X.shape[1]])
    print(dataset.X.shape)

    rollout_ticks_in_round = flatten_list(
        [[round_index * round_lengths.max_length_per_round + i for i in range(round_lengths.round_to_length[round_id])]
         for round_index, round_id in enumerate(round_lengths.round_ids)])
    rollout_tensor[rollout_ticks_in_round] = dataset.X

    src_first_tick_in_round = [tick_range.start for _, tick_range in round_lengths.round_to_subset_tick_indices.items()]
    similarity_tensor = dataset.similarity_tensor[src_first_tick_in_round].to(CUDA_DEVICE_STR)
    return rollout_tensor, similarity_tensor


PlayerEnableMask = Optional[torch.Tensor]


def step(rollout_tensor: torch.Tensor, all_similarity_tensor: torch.Tensor, pred_tensor: torch.Tensor,
         model: TransformerNestedHiddenLatentModel, round_lengths: RoundLengths, step_index: int, nav_data: NavData,
         player_enable_mask: PlayerEnableMask = None, fixed_pred: bool = False, convert_to_cpu: bool = True,
         save_new_pos: bool = True, pred_transformed: Optional[torch.Tensor] = None,
         pred_untransformed: Optional[torch.Tensor] = None):
    # skip rounds that are over, I know we have space, but wasteful as just going to filter out extra rows later
    # and cause problems as don't have non-computed input features (like visibility) at those time steps
    # and will crash open loop as everyone is 0
    rounds_containing_step_index = [step_index < round_lengths.round_to_length[round_lengths.round_ids[round_index]]
                                    for round_index in range(round_lengths.num_rounds)]
    rollout_tensor_input_indices = [step_index + round_lengths.max_length_per_round * round_index
                                    for round_index in range(round_lengths.num_rounds)
                                    if rounds_containing_step_index[round_index]]
    if len(rollout_tensor_input_indices) == 0:
        print('empty rollout tensor input indices')
    similarity_tensor = all_similarity_tensor[rounds_containing_step_index]
    rollout_tensor_output_indices = [index + 1 for index in rollout_tensor_input_indices]

    input_tensor = rollout_tensor[rollout_tensor_input_indices].to(CUDA_DEVICE_STR)
    input_pos_tensor = rearrange(input_tensor[:, model.players_pos_columns], 'b (p t d) -> b p t d',
                                 p=model.num_players, t=model.num_input_time_steps, d=model.num_dim)
    if not fixed_pred:
        temperature = torch.Tensor([1.]).to(CUDA_DEVICE_STR)
        pred = model(input_tensor, similarity_tensor, temperature)
        pred_prob = get_untransformed_outputs(pred)
        new_pred_tensor = rearrange(pred_prob, 'b p t d -> b (p t d)')
        if convert_to_cpu:
            pred_tensor[rollout_tensor_input_indices] = new_pred_tensor.to(CPU_DEVICE_STR)
        else:
            pred_tensor[rollout_tensor_input_indices] = new_pred_tensor
            # when want to save model outputs exactly
            if pred_transformed is not None:
                pred_transformed[rollout_tensor_input_indices] = get_transformed_outputs(pred)
            if pred_untransformed is not None:
                pred_untransformed[rollout_tensor_input_indices] = pred_prob
        nested_pred_labels = get_label_outputs(pred)
    else:
        nested_pred_labels = one_hot_max_to_index(
            rearrange(pred_tensor[rollout_tensor_input_indices], 'b (p t d) -> b p t d',
                      p=model.num_players, t=num_radial_ticks)).to(CUDA_DEVICE_STR)
    pred_labels = rearrange(nested_pred_labels, 'b p t d -> b (p t d)')

    # for rollout simulation, only want predictions for loss computation, no need to save last time step positions
    # also this will out of bounds in that case, as need prediction at time t and no position at t+1 to save at
    if save_new_pos:
        tmp_rollout = rollout_tensor[rollout_tensor_output_indices]
        new_player_pos = rearrange(compute_new_pos(input_pos_tensor, pred_labels, nav_data, False,
                                                   model.stature_to_speed_gpu), "b p t d -> b (p t d)")
        if convert_to_cpu:
            new_player_pos = new_player_pos.to(CPU_DEVICE_STR)
        if player_enable_mask is not None:
            tmp_rollout[:, model.players_pos_columns] = torch.where(player_enable_mask[rounds_containing_step_index],
                                                                    new_player_pos,
                                                                    tmp_rollout[:, model.players_pos_columns])
        else:
            tmp_rollout[:, model.players_pos_columns] = new_player_pos
        rollout_tensor[rollout_tensor_output_indices] = tmp_rollout


# undo the fixed length across all rounds, just get right length for each round
def match_round_lengths(df: pd.DataFrame, rollout_tensor: torch.Tensor, pred_tensor: torch.Tensor,
                        round_lengths: RoundLengths, cts: IOColumnTransformers) -> Tuple[pd.DataFrame, pd.DataFrame]:
    complete_matched_rollout_df = df.copy()
    required_indices = [round_lengths.round_id_to_list_id[round_id] * round_lengths.max_length_per_round + tick_index
                        for round_id, round_tick_range in round_lengths.round_to_tick_ids.items()
                        for tick_index in range(len(round_tick_range))]
    matched_rollout_tensor = rollout_tensor[required_indices]
    matched_pred_tensor = pred_tensor[required_indices]

    matched_rollout_df = cts.get_untransformed_values_whole_pd(matched_rollout_tensor, True)
    matched_pred_df = cts.get_untransformed_values_whole_pd(matched_pred_tensor, False)

    complete_matched_rollout_df[matched_rollout_df.columns] = matched_rollout_df

    return complete_matched_rollout_df, matched_pred_df


def delta_pos_rollout(loaded_model: LoadedModel):
    round_lengths = get_round_lengths(loaded_model.cur_loaded_df)
    rollout_tensor, similarity_tensor = build_rollout_and_similarity_tensors(round_lengths, loaded_model.cur_dataset)
    pred_tensor = torch.zeros(rollout_tensor.shape[0], loaded_model.cur_dataset.Y.shape[1])
    loaded_model.model.eval()
    with torch.no_grad():
        num_steps = round_lengths.max_length_per_round - 1
        with tqdm(total=num_steps, disable=False) as pbar:
            for step_index in range(num_steps):
                step(rollout_tensor, similarity_tensor, pred_tensor, loaded_model.model, round_lengths, step_index, nav_data)
                pbar.update(1)
    # need to modify cur_loaded_df as rollout_df has constant length of all rounds for sim efficiency
    loaded_model.cur_loaded_df, loaded_model.cur_inference_df = \
        match_round_lengths(loaded_model.cur_loaded_df, rollout_tensor, pred_tensor, round_lengths,
                            loaded_model.column_transformers)


load_data_options = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=False,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_all_human_data=True,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=True,
    #small_good_rounds=[small_human_good_rounds, all_human_28_second_filter_good_rounds],
    #similarity_dfs=[load_hdf5_to_pd(all_human_vs_small_human_similarity_hdf5_data_path),
    #                load_hdf5_to_pd(all_human_vs_human_28_similarity_hdf5_data_path)],
    limit_by_similarity=False,
    train_test_split_file_name=train_test_split_file_name
)

nav_data = None

if __name__ == "__main__":
    nav_data = NavData(CUDA_DEVICE_STR)

    load_data_options.custom_limit_fn = limit_to_every_nth_row
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
    vis(loaded_model, delta_pos_rollout, " Simulator")
