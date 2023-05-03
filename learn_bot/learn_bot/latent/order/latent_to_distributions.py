from typing import List

import torch

from learn_bot.latent.engagement.latent_to_distributions import add_to_zeros, zero_adjustment_amount

num_target_options = 6

def get_order_probability(latent_tensor: torch.Tensor, observation: torch.Tensor, col_ranges: List[range]):
    per_player_order_probs = latent_tensor * observation
    # remove negatives for dead players, don't count those during loss
    # 0.999 + 0.001 below will send to 1 so have no impact on loss
    negs_removed = torch.where(per_player_order_probs >= 0., per_player_order_probs, 1 - zero_adjustment_amount)
    return -1 * torch.sum(torch.log(add_to_zeros(torch.concat([negs_removed], dim=1))))
