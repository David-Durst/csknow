import torch

num_aggression_options = 3

def get_aggression_distributions(latent_tensor: torch.Tensor):
    fire_next_2s_distribution = torch.concat(
        [latent_tensor[:, [0]], torch.sum(latent_tensor[:, 1:num_aggression_options], dim=1, keepdim=True)], dim=1)
    visible_next_2s_distribution = fire_next_2s_distribution
    nearest_enemy_change = latent_tensor
    return torch.concat([fire_next_2s_distribution, visible_next_2s_distribution, nearest_enemy_change], dim=1)
