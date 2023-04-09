from typing import List

import torch

num_target_options = 6

def get_order_probability(latent_tensor: torch.Tensor, observation: torch.Tensor, col_ranges: List[range]):
    #nested_observation = torch.unflatten(observation, 1, torch.Size([10, 6]))
    per_player_order_probs = latent_tensor * observation
    #flattened_probs = torch.flatten(per_player_order_probs, start_dim=1)
    # remove negatives for dead players, don't count those during loss
    # 0.999 + 0.001 below will send to 1 so have no impact on loss
    negs_removed = torch.where(per_player_order_probs >= 0., per_player_order_probs, 0.999)
    # add 0.001 so that no probs are 0
    return -1 * \
        torch.sum(torch.log(0.001 + torch.concat([negs_removed], dim=1)))
