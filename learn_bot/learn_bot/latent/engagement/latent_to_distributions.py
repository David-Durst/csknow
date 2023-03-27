import torch

num_target_options = 6

def get_engagement_target_distributions(latent_tensor: torch.Tensor):
    position_distribution = torch.concat(
        [torch.sum(latent_tensor[:, 0:num_target_options-1], dim=1, keepdim=True),
         latent_tensor[:, [num_target_options-1]]], dim=1)
    aim_distribution = position_distribution
    nearest_enemy_distribution = latent_tensor
    return torch.concat([position_distribution, aim_distribution, nearest_enemy_distribution], dim=1)

