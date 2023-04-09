from typing import List

import torch

num_target_options = 6

def get_engagement_target_distributions(latent_tensor: torch.Tensor):
    position_distribution = torch.concat(
        [torch.sum(latent_tensor[:, 0:num_target_options-1], dim=1, keepdim=True),
         latent_tensor[:, [num_target_options-1]]], dim=1)
    aim_distribution = position_distribution
    nearest_enemy_distribution = latent_tensor
    return torch.concat([position_distribution, aim_distribution, nearest_enemy_distribution], dim=1)


zero_adjustment_amount = 0.001
def add_to_zeros(input_tensor: torch.Tensor):
     return torch.where(input_tensor != 0., input_tensor, zero_adjustment_amount)

def get_engagement_probability(latent_tensor: torch.Tensor, observation: torch.Tensor, col_ranges: List[range]):
    #position_prob = torch.concat(
    #    [latent_tensor[:, [num_target_options-1]] * observation[:, [col_ranges[0][0]]],
    #     latent_tensor[:, 0:num_target_options-1] * observation[:, [col_ranges[0][1] for _ in range(num_target_options-1)]]],
    #    dim=1)
    aim_prob = torch.concat(
        [latent_tensor[:, [num_target_options-1]] * observation[:, [col_ranges[1][0]]],
         latent_tensor[:, 0:num_target_options-1] * observation[:, [col_ranges[1][1] for _ in range(num_target_options-1)]]],
        dim=1)
    nearest_enemy_change = latent_tensor * observation[:, col_ranges[2]]
    # add 0.001 so that no probs are 0
    return -1 * \
        torch.sum(torch.log(add_to_zeros(torch.concat([nearest_enemy_change], dim=1))))
