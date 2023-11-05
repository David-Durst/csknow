import glob
from pathlib import Path
from typing import List

import pandas as pd


metrics_summary_csv_name = "metrics_summary.csv"


def load_metrics(metric_dir: Path) -> pd.DataFrame:
    partial_dfs: List[pd.DataFrame] = []
    for f in glob.glob(metric_dir / '*.csv'):
        f_path = Path(f)
        if metrics_summary_csv_name == f_path.name:
            continue
        partial_dfs.append(pd.read_csv(f))
    return pd.concat(partial_dfs)


def compare_human_bot_metrics(human_dir: Path, bot_dir: Path):
    human_metrics_df = load_metrics(human_dir)
    bot_metrics_df = load_metrics(human_dir)

