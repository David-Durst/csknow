import torch

from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.place_area.simulation.simulator import RoundLengths


def update_nn_position_rollout_tensor(loaded_model: LoadedModel, round_lengths: RoundLengths,
                                      ground_truth_rollout_tensor: torch.Tensor) -> torch.Tensor:
    raise NotImplemented