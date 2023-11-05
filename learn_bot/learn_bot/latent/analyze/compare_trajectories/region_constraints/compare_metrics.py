import glob
from pathlib import Path
from typing import List, Dict
from math import pow

import pandas as pd
import scipy

from learn_bot.latent.analyze.compare_trajectories.region_constraints.compute_constraint_metrics import \
    test_name_column, percent_constraint_valid_column
from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path


t_tests_csv = "t_tests.csv"


def load_metrics(metric_dir: Path) -> pd.DataFrame:
    partial_dfs: List[pd.DataFrame] = []
    for f in glob.glob(str(metric_dir / '*.csv')):
        f_path = Path(f)
        if t_tests_csv == f_path.name:
            continue
        partial_dfs.append(pd.read_csv(f))
    return pd.concat(partial_dfs)


tests_to_compare: List[List[str]] = [
    ["AttackASpawnTLong", "AttackASpawnTLongTwoTeammates", "AttackASpawnTExtendedA"],
    ["AttackBDoorsTeammateHole", "AttackBHoleTeammateBDoors"],
    ["DefendACTCat", "DefendACTCatTwoTeammates"],
    ["DefendACTLong", "DefendACTLongWithTeammate", "DefendACTLongWithTwoTeammates"],
    ["DefendBCTSite", "DefendBCTTuns"],
    ["DefendBCTHole", "DefendBCTHoleTwoTeammates"]
]


base_test_name_column = 'Base Test'
base_mean_time_valid_column = 'Base Mean Time Valid'
base_std_time_valid_column = 'Base Std Time Valid'
other_test_name_column = 'Other Test'
other_mean_time_valid_column = 'Other Mean Time Valid'
other_std_time_valid_column = 'Other Std Time Valid'
t_column = 'T Score'


def run_t_test(data_dir: Path):
    metrics_df = load_metrics(data_dir)
    t_dicts: List[Dict] = []
    for test_group in tests_to_compare:
        base_test_name = test_group[0]
        base_percent_constraint_valid = metrics_df
        base_row = metrics_df[metrics_df[test_name_column] == base_test_name].iloc[0]
        # https://www.bmj.com/about-bmj/resources-readers/publications/statistics-square-one/7-t-tests
        for other_test_name in test_group[1:]:
            other_row = metrics_df[metrics_df[test_name_column] == other_test_name].iloc[0]
            n1s1 = (base_row[num_trials_column] - 1) * pow(base_row[std_time_constraint_valid_column], 2.)
            n2s2 = (other_row[num_trials_column] - 1) * pow(other_row[std_time_constraint_valid_column], 2.)
            sp = (n1s1 + n2s2) / (base_row[num_trials_column] + other_row[num_trials_column] - 2)
            se = pow(sp, 0.5) * (1. / base_row[num_trials_column] + 1. / other_row[num_trials_column])
            t = abs(base_row[mean_time_constraint_valid_column] - other_row[mean_time_constraint_valid_column]) / se
            t_dicts.append({
                base_test_name_column: base_test_name,
                base_mean_time_valid_column: base_row[mean_time_constraint_valid_column],
                base_std_time_valid_column: base_row[std_time_constraint_valid_column],
                other_test_name_column: other_test_name,
                other_mean_time_valid_column: other_row[mean_time_constraint_valid_column],
                other_std_time_valid_column: other_row[std_time_constraint_valid_column],
                t_column: t
            })
    t_df = pd.DataFrame.from_records(t_dicts)
    t_df.to_csv(data_dir / t_tests_csv)


if __name__ == "__main__":
    human_knn_dir = similarity_plots_path / 'human_knn' / 'matches_20_restrict_future_True'
    run_t_test(human_knn_dir)
    bot_11_4 = similarity_plots_path / '_11_4_23_prebaked_no_mask_100_tries'
    run_t_test(bot_11_4)
    bot_11_2 = similarity_plots_path / '_11_2_23_prebaked_no_mask_100_tries'
    run_t_test(bot_11_2)


