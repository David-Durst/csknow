from typing import List

import torch

num_aggression_options = 3

def get_aggression_distributions(latent_tensor: torch.Tensor):
    fire_next_2s_distribution = torch.concat(
        [latent_tensor[:, [0]], torch.sum(latent_tensor[:, 1:num_aggression_options], dim=1, keepdim=True)], dim=1)
    visible_next_2s_distribution = fire_next_2s_distribution
    nearest_enemy_change = latent_tensor
    return torch.concat([fire_next_2s_distribution, visible_next_2s_distribution, nearest_enemy_change], dim=1)

def get_aggression_probability(latent_tensor: torch.Tensor, observation: torch.Tensor, col_ranges: List[range]):
    fire_next_2s_prob = torch.concat(
        [latent_tensor[:, [0]] * observation[:, [col_ranges[0][0]]],
         latent_tensor[:, [1]] * observation[:, [col_ranges[0][1]]],
         latent_tensor[:, [2]] * observation[:, [col_ranges[0][1]]]],
        dim=1)
    visible_next_2s_prob = torch.concat(
        [latent_tensor[:, [0]] * observation[:, [col_ranges[1][0]]],
         latent_tensor[:, [1]] * observation[:, [col_ranges[1][1]]],
         latent_tensor[:, [2]] * observation[:, [col_ranges[1][1]]]],
        dim=1)
    nearest_enemy_change = latent_tensor * observation[:, col_ranges[2]]
    #
    # add 0.001 so that no probs are 0
    return -1 * \
        torch.sum(torch.log(0.001 + torch.concat([nearest_enemy_change], dim=1)))
