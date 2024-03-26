import sys
from typing import Optional

import pandas as pd

from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path


def aggregate_trajectory_events(rollout_extensions: list[str], rollout_prefix: str):
    for i, rollout_extension in enumerate(rollout_extensions):
        plots_path = similarity_plots_path / (rollout_prefix + rollout_extension)
        new_offense_flanks_df = pd.read_csv(plots_path / "offense_flanks_pct_for_aggregation.csv", index_col=0)
        new_defense_spread_df = pd.read_csv(plots_path / "defense_spread_pct_for_aggregation.csv", index_col=0)
        new_mistakes_df = pd.read_csv(plots_path / "mistakes_aggregation.csv", index_col=0)
        if i == 0:
            offense_flanks_df = new_offense_flanks_df
            defense_spread_df = new_defense_spread_df
            mistakes_df = new_mistakes_df
        else:
            new_offense_flanks_df.columns = offense_flanks_df.columns
            offense_flanks_df = offense_flanks_df + new_offense_flanks_df
            new_defense_spread_df.columns = defense_spread_df.columns
            defense_spread_df = defense_spread_df + new_defense_spread_df
            new_mistakes_df.columns = mistakes_df.columns
            mistakes_df = mistakes_df + new_mistakes_df
    x = 1


if __name__ == "__main__":
    rollout_extensions = sys.argv[1].split(',')
    rollout_prefix = sys.argv[2] if len(sys.argv) >= 3 else ""
    aggregate_trajectory_events(rollout_extensions, rollout_prefix)