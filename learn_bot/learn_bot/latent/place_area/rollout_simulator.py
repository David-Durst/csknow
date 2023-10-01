from dataclasses import dataclass
from random import random
from typing import Optional, Tuple
import torch
from einops import rearrange

from learn_bot.latent.latent_hdf5_dataset import round_id_index_in_tensor
from learn_bot.latent.place_area.simulator import step, get_round_lengths, RoundLengths
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel


# build the minimum RoundLengths necessary for simulator step function
def get_rollout_round_lengths(indices: torch.Tensor) -> RoundLengths:
    same_round = indices[:, [0], round_id_index_in_tensor] == indices[:, :, round_id_index_in_tensor]
    # assumes same round until not in same round, never go back to same round
    # this can only go bad if each hdf5 file is so short that you can go from one to next and get back to original
    # round id, which isn't possible for 2 hour long hdf5 ids and 5 second sequences
    length_per_round = same_round.sum(axis=1)
    return RoundLengths(indices.shape[0], indices.shape[1], list(range(indices.shape[0])),
                        {}, {}, {i: length_per_round[i] for i in range(indices.shape[0])}, {}, True, {})


@dataclass
class RolloutBatchResult:
    model_pred_flattened: Tuple[torch.Tensor, torch.Tensor]
    Y_flattened: torch.Tensor
    duplicated_last_flattened: torch.Tensor


def rollout_simulate(X: torch.Tensor, Y: torch.Tensor, similarity: torch.Tensor, duplicated_last: torch.Tensor,
                     indices: torch.Tensor, model: TransformerNestedHiddenLatentModel,
                     percent_steps_predicted: float) -> RolloutBatchResult:
    round_lengths = get_rollout_round_lengths(indices)
    valid_flattened_indices = []
    round_start_index = 0
    for round_id in round_lengths.round_ids:
        valid_flattened_indices += list(range(round_start_index,
                                              round_start_index + round_lengths.round_to_length[round_id]))
        round_start_index += round_lengths.max_length_per_round

    X_flattened = rearrange(X, 'b t d -> (b t) d')
    Y_flattened = rearrange(Y, 'b t d -> (b t) d')
    # step assumes one similarity row per round, so just take first row per round
    similarity_flattened = similarity[:, 0, :]
    pred_flattened = torch.zeros(X_flattened.shape[0], Y.shape[2], dtype=Y.dtype, device=X.device)
    pred_transformed = torch.zeros(X_flattened.shape[0], model.num_players, model.num_output_time_steps,
                                   Y.shape[2] // model.num_players // model.num_output_time_steps,
                                   dtype=X.dtype, device=Y.device)
    pred_untransformed = torch.zeros(X_flattened.shape[0], model.num_players, model.num_output_time_steps,
                                     Y.shape[2] // model.num_players // model.num_output_time_steps,
                                     dtype=Y.dtype, device=Y.device)
    duplicated_last_flattened = rearrange(duplicated_last, 'b t -> (b t)')

    had_random = False
    for i in range(round_lengths.max_length_per_round):
        # if got to end and didn't have a random yet, make sure to make one random so have something to back prop
        if random() <= percent_steps_predicted or (not had_random and i == round_lengths.max_length_per_round - 1):
            step(X_flattened, similarity_flattened, pred_flattened, model, round_lengths, i, model.nav_data_cuda,
                 convert_to_cpu=False, save_new_pos=i < (round_lengths.max_length_per_round - 1),
                 pred_untransformed=pred_untransformed, pred_transformed=pred_transformed)
            had_random = True
        else:
            step_flattened_indices = [round_index * round_lengths.max_length_per_round + i
                                      for round_index in range(round_lengths.num_rounds)]
            pred_flattened[step_flattened_indices] = Y[:, i, :]

    return RolloutBatchResult((pred_transformed[valid_flattened_indices], pred_untransformed[valid_flattened_indices]),
                              Y_flattened[valid_flattened_indices],
                              duplicated_last_flattened[valid_flattened_indices])


