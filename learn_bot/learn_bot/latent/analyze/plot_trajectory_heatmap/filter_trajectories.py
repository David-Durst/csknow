from dataclasses import dataclass
from typing import Set, Optional

import numpy as np
import pandas as pd

from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import AABB


@dataclass
class TrajectoryFilterOptions:
    valid_round_ids: Optional[Set[int]] = None
    trajectory_counter: Optional[pd.Series] = None
    player_starts_in_region: Optional[AABB] = None
    round_game_seconds: Optional[int] = None


default_trajectory_filter_options = TrajectoryFilterOptions()
