import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from tqdm import tqdm

from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import round_id_column, tick_id_column
from learn_bot.latent.order.column_names import delta_pos_grid_num_cells_per_dim, delta_pos_grid_cell_dim
from learn_bot.latent.train import manual_latent_team_hdf5_data_path, rollout_latent_team_hdf5_data_path, \
    latent_team_hdf5_data_path
from learn_bot.latent.transformer_nested_hidden_latent_model import *
from learn_bot.latent.vis.run_vis_checkpoint import load_model_file_for_rollout
from learn_bot.latent.vis.vis import vis
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.io_transforms import get_untransformed_outputs, CPU_DEVICE_STR


@dataclass
class RoundLengths:
    num_rounds: int
    max_length_per_round: int
    round_to_tick_ids: Dict[int, range]


def get_round_lengths(df: pd.DataFrame) -> RoundLengths:
    grouped_df = df.groupby([round_id_column]).agg({tick_id_column: ['count', 'min', 'max']})
    result = RoundLengths(len(grouped_df), max(grouped_df[tick_id_column]['count']), {})
    for round_id, round_row in grouped_df[tick_id_column].iterrows():
        result.round_to_tick_ids[round_id] = range(round_row['min'], round_row['max']+1)
    return result


# src tensor is variable length per round, rollout tensor is fixed length for efficiency
def build_rollout_tensor(round_lengths: RoundLengths, dataset: LatentDataset) -> torch.Tensor:
    result = torch.zeros([round_lengths.num_rounds * (round_lengths.max_length_per_round + 1), dataset.X.shape[1]])
    src_first_tick_in_round = [tick_range.start for _, tick_range in round_lengths.round_to_tick_ids.items()]
    rollout_first_tick_in_round = [round_index * round_lengths.max_length_per_round
                               for round_index in range(round_lengths.num_rounds)]
    result[rollout_first_tick_in_round] = dataset.X[src_first_tick_in_round]
    return result


def step(rollout_tensor: torch.Tensor, pred_tensor: torch.Tensor, model: TransformerNestedHiddenLatentModel,
         round_lengths: RoundLengths, step_index: int):
    rollout_tensor_input_indices = [step_index + round_lengths.max_length_per_round * round_id
                                    for round_id in range(round_lengths.num_rounds)]
    rollout_tensor_output_indices = [index + 1 for index in rollout_tensor_input_indices]

    pred = get_untransformed_outputs(model(rollout_tensor[rollout_tensor_input_indices].to(CUDA_DEVICE_STR)))
    pred_tensor[rollout_tensor_input_indices] = pred.to(CPU_DEVICE_STR)
    pred_per_player = torch.argmax(rearrange(pred, 'b (p d) -> b p d', p=len(specific_player_place_area_columns)), 2)

    x_index = torch.floor(torch.remainder(pred_per_player, delta_pos_grid_num_cells_per_dim)) - \
              int(delta_pos_grid_num_cells_per_dim / 2)
    y_index = torch.floor(pred_per_player / delta_pos_grid_num_cells_per_dim) - int(delta_pos_grid_num_cells_per_dim / 2)
    z_index = torch.zeros_like(x_index)
    pos_index = rearrange(torch.stack([x_index, y_index, z_index], dim=-1), 'b p d -> b (p d)')
    unscaled_pos_change = (pos_index * delta_pos_grid_cell_dim)

    tmp_rollout_tensor = rollout_tensor[rollout_tensor_input_indices]
    tmp_rollout_tensor[:, model.players_pos_columns] += (pos_index * delta_pos_grid_cell_dim).to(CPU_DEVICE_STR)
    rollout_tensor[rollout_tensor_output_indices] = tmp_rollout_tensor


def match_round_lengths(df: pd.DataFrame, rollout_tensor: torch.Tensor, pred_tensor: torch.Tensor,
                        round_lengths: RoundLengths, cts: IOColumnTransformers) -> Tuple[pd.DataFrame, pd.DataFrame]:
    complete_matched_rollout_df = df.copy()
    required_indices = [tick_index for _, round_tick_range in round_lengths.round_to_tick_ids.items()
                        for tick_index in round_tick_range]
    matched_rollout_tensor = rollout_tensor[required_indices]
    matched_pred_tensor = pred_tensor[required_indices]

    matched_rollout_df = cts.get_untransformed_values_whole_pd(matched_rollout_tensor, True)
    matched_pred_df = cts.get_untransformed_values_whole_pd(matched_pred_tensor, False)

    complete_matched_rollout_df[matched_rollout_df.columns] = matched_rollout_df

    return complete_matched_rollout_df, matched_pred_df


def delta_pos_rollout(df: pd.DataFrame, dataset: LatentDataset, model: TransformerNestedHiddenLatentModel,
                      cts: IOColumnTransformers) -> Tuple[pd.DataFrame, pd.DataFrame]:
    round_lengths = get_round_lengths(df)
    rollout_tensor = build_rollout_tensor(round_lengths, dataset)
    pred_tensor = torch.zeros(rollout_tensor.shape[0], dataset.Y.shape[1])
    model.eval()
    with torch.no_grad():
        num_steps = round_lengths.max_length_per_round
        with tqdm(total=num_steps, disable=False) as pbar:
            for step_index in range(num_steps):
                step(rollout_tensor, pred_tensor, model, round_lengths, step_index)
                pbar.update(1)
    return match_round_lengths(df, rollout_tensor, pred_tensor, round_lengths, cts)


manual_data = True
rollout_data = False

if __name__ == "__main__":
    if manual_data:
        all_data_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path)
        #all_data_df = all_data_df[all_data_df['test name'] == b'LearnedGooseToCatScript']
    elif rollout_data:
        all_data_df = load_hdf5_to_pd(rollout_latent_team_hdf5_data_path)
    else:
        all_data_df = load_hdf5_to_pd(latent_team_hdf5_data_path)
        all_data_df = all_data_df[(all_data_df['valid'] == 1.) & (all_data_df['c4 status'] < 2)]
    all_data_df = all_data_df.copy()

    load_result = load_model_file_for_rollout(all_data_df, "delta_pos_checkpoint.pt")

    rollout_df, pred_df = delta_pos_rollout(load_result.test_df, load_result.test_dataset, load_result.model,
                                            load_result.column_transformers)
    vis(rollout_df, pred_df)
