from dataclasses import dataclass
from typing import Set, Optional

import numpy as np
import pandas as pd


@dataclass
class TrajectoryFilterOptions:
    valid_round_ids: Optional[Set[int]] = None
