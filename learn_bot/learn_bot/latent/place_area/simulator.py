from pathlib import Path

import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from tqdm import tqdm

from learn_bot.latent.analyze.comparison_column_names import small_human_good_rounds, \
    all_human_28_second_filter_good_rounds, all_human_vs_small_human_similarity_hdf5_data_path, \
    all_human_vs_human_28_similarity_hdf5_data_path
from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import round_id_column, tick_id_column
from learn_bot.latent.load_model import load_model_file, LoadedModel
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import delta_pos_grid_num_cells_per_xy_dim, \
    delta_pos_grid_cell_dim, \
    delta_pos_grid_num_xy_cells_per_z_change, compute_new_pos, NavData
from learn_bot.latent.place_area.load_data import human_latent_team_hdf5_data_path, manual_latent_team_hdf5_data_path, \
    rollout_latent_team_hdf5_data_path, LoadDataResult, LoadDataOptions
from learn_bot.latent.transformer_nested_hidden_latent_model import *
from learn_bot.latent.vis.vis import vis
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd, load_hdf5_extra_to_list
from learn_bot.libs.io_transforms import get_untransformed_outputs, CPU_DEVICE_STR, get_label_outputs


@dataclass
class RoundLengths:
    num_rounds: int
    max_length_per_round: int
    round_to_tick_ids: Dict[int, range]
    round_id_to_list_id: Dict[int, int]


def get_round_lengths(df: pd.DataFrame) -> RoundLengths:
    grouped_df = df.groupby([round_id_column]).agg({tick_id_column: ['count', 'min', 'max']})
    result = RoundLengths(len(grouped_df), max(grouped_df[tick_id_column]['count']), {}, {})
    list_id = 0
    for round_id, round_row in grouped_df[tick_id_column].iterrows():
        result.round_to_tick_ids[round_id] = range(round_row['min'], round_row['max']+1)
        result.round_id_to_list_id[round_id] = list_id
        list_id += 1
    return result


# src tensor is variable length per round, rollout tensor is fixed length for efficiency
def build_rollout_and_similarity_tensors(round_lengths: RoundLengths, dataset: LatentDataset) -> \
        Tuple[torch.Tensor,torch.Tensor]:
    rollout_tensor = torch.zeros([round_lengths.num_rounds * round_lengths.max_length_per_round, dataset.X.shape[1]])
    src_first_tick_in_round = [tick_range.start for _, tick_range in round_lengths.round_to_tick_ids.items()]
    rollout_first_tick_in_round = [round_index * round_lengths.max_length_per_round
                                   for round_index in range(round_lengths.num_rounds)]
    rollout_tensor[rollout_first_tick_in_round] = dataset.X[src_first_tick_in_round]
    similarity_tensor = dataset.similarity_tensor[src_first_tick_in_round].to(CUDA_DEVICE_STR)
    similarity_tensor[:, :] = 1.
    return rollout_tensor, similarity_tensor


def step(rollout_tensor: torch.Tensor, similarity_tensor: torch.Tensor, pred_tensor: torch.Tensor,
         model: TransformerNestedHiddenLatentModel, round_lengths: RoundLengths, step_index: int, nav_data: NavData):
    rollout_tensor_input_indices = [step_index + round_lengths.max_length_per_round * round_id
                                    for round_id in range(round_lengths.num_rounds)]
    rollout_tensor_output_indices = [index + 1 for index in rollout_tensor_input_indices]

    input_tensor = rollout_tensor[rollout_tensor_input_indices].to(CUDA_DEVICE_STR)
    input_pos_tensor = rearrange(input_tensor[:, model.players_pos_columns], 'b (p t d) -> b p t d',
                                 p=model.num_players, t=model.num_time_steps, d=model.num_dim)
    pred = model(input_tensor, similarity_tensor)
    pred_prob = get_untransformed_outputs(pred)
    pred_tensor[rollout_tensor_input_indices] = pred_prob.to(CPU_DEVICE_STR)
    pred_labels = get_label_outputs(pred)

    tmp_rollout = rollout_tensor[rollout_tensor_input_indices]
    tmp_rollout[:, model.players_pos_columns] = \
        rearrange(compute_new_pos(input_pos_tensor, pred_labels, nav_data, False,
                                  model.stature_to_speed_gpu).to(CPU_DEVICE_STR), "b p t d -> b (p t d)")
    rollout_tensor[rollout_tensor_output_indices] = tmp_rollout


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
    small_good_rounds=[small_human_good_rounds, all_human_28_second_filter_good_rounds],
    similarity_dfs=[load_hdf5_to_pd(all_human_vs_small_human_similarity_hdf5_data_path),
                    load_hdf5_to_pd(all_human_vs_human_28_similarity_hdf5_data_path)],
    limit_by_similarity=False
)

nav_data = None

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

    loaded_model = load_model_file(load_data_result)
    vis(loaded_model, delta_pos_rollout)
