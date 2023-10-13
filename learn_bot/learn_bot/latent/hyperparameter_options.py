from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import data_ticks_per_second, data_ticks_per_sim_tick
from learn_bot.latent.train_paths import checkpoints_path
from learn_bot.latent.transformer_nested_hidden_latent_model import PlayerMaskType

from datetime import datetime

now = datetime.now()
now_str = now.strftime("%m_%d_%Y__%H_%M_%S")

@dataclass
class HyperparameterOptions:
    bc_epochs: int = 25
    probabilistic_rollout_epochs: int = 0
    full_rollout_epochs: int = 0
    batch_size: int = 1024
    num_input_time_steps: int = 1
    learning_rate: float = 4e-5
    weight_decay: float = 0.
    layers: int = 2
    heads: int = 4
    noise_var: float = 20.
    rollout_seconds: Optional[float] = 2.
    player_mask_type: PlayerMaskType = PlayerMaskType.NoMask
    weight_not_move_loss: Optional[float] = None
    drop_history_probability: Optional[float] = None
    comment: str = ""

    def __str__(self):
        return f"{now_str}_bc_{self.bc_epochs}_pr_{self.probabilistic_rollout_epochs}_fr_{self.full_rollout_epochs}_" \
               f"b_{self.batch_size}_it_{self.num_input_time_steps}_" \
               f"lr_{self.learning_rate}_wd_{self.weight_decay}_" \
               f"l_{self.layers}_h_{self.heads}_n_{self.noise_var}_" \
               f"ros_{self.rollout_seconds}_m_{str(self.player_mask_type)}_" \
               f"w_{self.weight_not_move_loss}_dh_{self.drop_history_probability}_c_{self.comment}"

    def get_checkpoints_path(self) -> Path:
        return checkpoints_path / str(self)

    def num_epochs(self) -> int:
        return self.bc_epochs + self.probabilistic_rollout_epochs + self.full_rollout_epochs

    def percent_rollout_steps_predicted(self, epoch_num: int) -> float:
        if epoch_num < self.bc_epochs:
            return 0.
        elif epoch_num >= self.bc_epochs + self.probabilistic_rollout_epochs:
            return 1.
        else:
            return (epoch_num - self.bc_epochs) / self.probabilistic_rollout_epochs

    def get_rollout_steps(self, epoch_num: int):
        assert epoch_num < self.num_epochs()
        percent_rollout_steps_predicted = self.percent_rollout_steps_predicted(epoch_num)
        # 1 for behavior cloning if no rollout
        if self.rollout_seconds is None or percent_rollout_steps_predicted == 0.:
            return 1
        else:
            return int(data_ticks_per_second / data_ticks_per_sim_tick * self.rollout_seconds)
