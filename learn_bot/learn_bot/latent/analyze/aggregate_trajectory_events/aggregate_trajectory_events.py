import sys
from pathlib import Path
from typing import Optional, List

from einops import rearrange
import numpy as np
import pandas as pd
from scipy.stats import iqr

from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path


def aggregate_one_event(rollout_extensions: list[str], rollout_prefix: str, event_csv_path: Path,
                        diff_to_first_player_type: bool):
    event_nps: List[np.ndarray] = []

    for i, rollout_extension in enumerate(rollout_extensions):
        plots_path = similarity_plots_path / (rollout_prefix + rollout_extension)
        new_event_df = pd.read_csv(plots_path / event_csv_path, index_col=0)
        if i == 0:
            column_names = new_event_df.columns
            row_index = new_event_df.index
        event_nps.append(new_event_df.values)

    event_np = np.stack(event_nps, axis=-1)

    if diff_to_first_player_type:
        column_names = column_names[1:]
        event_np = np.abs((event_np[:, 1:, :] - event_np[:, [0], :]) / event_np[:, [0], :])

    per_event_median_np = np.median(event_np, axis=2)
    per_event_median_df = pd.DataFrame(data=per_event_median_np, index=row_index, columns=column_names)
    per_event_iqr_np = iqr(event_np, axis=2)
    per_event_iqr_df = pd.DataFrame(data=per_event_iqr_np, index=row_index, columns=column_names)

    aggregation_plots_path = similarity_plots_path / ("agg_" + rollout_prefix + rollout_extension)

    # in order to aggregate across different events, get player type first dimension,
    # then event type, then different trials of events
    player_event_trials_np = rearrange('i j k -> j (i k)', event_np)
    all_events_median_np = np.median(event_np, axis=1)
    all_events_median_series = pd.Series(data=all_events_median_np, index=column_names)
    all_events_iqr_np = iqr(event_np, axis=1)
    all_events_median_series = pd.Series(data=all_events_iqr_np, index=column_names)


def aggregate_trajectory_events(rollout_extensions: list[str], rollout_prefix: str):
    aggregate_one_event(rollout_extensions, rollout_prefix, Path("offense_flanks_pct_for_aggregation.csv"), True)
    aggregate_one_event(rollout_extensions, rollout_prefix, Path("defense_spread_pct_for_aggregation.csv"), True)
    aggregate_one_event(rollout_extensions, rollout_prefix, Path("mistakes_aggregation.csv"), True)
    aggregate_one_event(rollout_extensions, rollout_prefix, Path("diff") / "emd_no_filter.txt", False)
    aggregate_one_event(rollout_extensions, rollout_prefix, Path("diff") / "emd_only_kill.txt", False)


if __name__ == "__main__":
    rollout_extensions = sys.argv[1].split(',')
    rollout_prefix = sys.argv[2] if len(sys.argv) >= 3 else ""
    aggregate_trajectory_events(rollout_extensions, rollout_prefix)