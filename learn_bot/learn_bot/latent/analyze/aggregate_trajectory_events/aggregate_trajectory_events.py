import sys
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.stats import stats

from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path


def aggregate_trajectory_events(rollout_extensions: list[str], rollout_prefix: str):
    offense_flanks_nps: List[np.ndarray] = []
    defense_spread_nps: List[np.ndarray] = []
    mistakes_nps: List[np.ndarray] = []

    for i, rollout_extension in enumerate(rollout_extensions):
        plots_path = similarity_plots_path / (rollout_prefix + rollout_extension)
        new_offense_flanks_df = pd.read_csv(plots_path / "offense_flanks_pct_for_aggregation.csv", index_col=0)
        new_defense_spread_df = pd.read_csv(plots_path / "defense_spread_pct_for_aggregation.csv", index_col=0)
        new_mistakes_df = pd.read_csv(plots_path / "mistakes_aggregation.csv", index_col=0)

        if i == 0:
            offense_flanks_df = new_offense_flanks_df
            defense_spread_df = new_defense_spread_df
            mistakes_df = new_mistakes_df

        offense_flanks_nps.append(new_offense_flanks_df.values)
        defense_spread_nps.append(new_defense_spread_df.values)
        mistakes_nps.append(new_mistakes_df.values)

    offense_flanks_np = np.stack(offense_flanks_nps, axis=-1)
    defense_spread_np = np.stack(defense_spread_nps, axis=-1)
    mistakes_np = np.stack(mistakes_nps, axis=-1)

    offense_flanks_median_np = np.median(offense_flanks_np, axis=2)
    defense_spread_median_np = np.median(defense_spread_np, axis=2)
    mistakes_median_np = np.median(mistakes_np, axis=2)

    offense_flanks_iqr_np = stats.iqr(offense_flanks_np, axis=2)
    defense_spread_iqr_np = np.iqr(defense_spread_np, axis=2)
    mistakes_iqr_np = np.iqr(mistakes_np, axis=2)
    x = 1


if __name__ == "__main__":
    rollout_extensions = sys.argv[1].split(',')
    rollout_prefix = sys.argv[2] if len(sys.argv) >= 3 else ""
    aggregate_trajectory_events(rollout_extensions, rollout_prefix)