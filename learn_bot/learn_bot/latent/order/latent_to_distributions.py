from typing import List

import torch

num_target_options = 6

def get_order_probability(latent_tensor: torch.Tensor, observation: torch.Tensor, col_ranges: List[range]):
    per_player_order_probs = latent_tensor * observation
    flattened_probs = torch.flatten(per_player_order_probs, start_dim=1)
    # add 0.001 so that no probs are 0
    return -1 * \
        torch.sum(torch.log(0.001 + torch.concat([flattened_probs], dim=1)))
