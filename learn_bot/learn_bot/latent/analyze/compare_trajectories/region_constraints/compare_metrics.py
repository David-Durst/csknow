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


def load_metrics(metric_dir: Path) -> Dict[str, pd.DataFrame]:
    result_dfs: Dict[str, pd.DataFrame] = {}
    for f in glob.glob(str(metric_dir / '*.csv')):
        f_path = Path(f)
        if t_tests_csv == f_path.name:
            continue
        df = pd.read_csv(f, index_col=False)
        result_dfs[df.iloc[0][test_name_column]] = df
    return result_dfs


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
p_column = 'P Value'


def run_t_test(data_dir: Path) -> pd.DataFrame:
    print(f'processing {data_dir}')
    metrics_dfs = load_metrics(data_dir)
    t_dicts: List[Dict] = []
    for test_group in tests_to_compare:
        base_test_name = test_group[0]
        base_percents = metrics_dfs[base_test_name][percent_constraint_valid_column]
        base_percents_mean = base_percents.mean()
        base_percents_std = base_percents.std()
        # https://www.bmj.com/about-bmj/resources-readers/publications/statistics-square-one/7-t-tests
        for other_test_name in test_group[1:]:
            other_percents = metrics_dfs[other_test_name][percent_constraint_valid_column]
            other_percents_mean = other_percents.mean()
            other_percents_std = other_percents.std()
            t_test_result = scipy.stats.ttest_ind(base_percents, other_percents)
            t_dicts.append({
                base_test_name_column: base_test_name,
                base_mean_time_valid_column: base_percents_mean,
                base_std_time_valid_column: base_percents_std,
                other_test_name_column: other_test_name,
                other_mean_time_valid_column: other_percents_mean,
                other_std_time_valid_column: other_percents_std,
                t_column: t_test_result.statistic,
                p_column: t_test_result.pvalue
            })
    t_df = pd.DataFrame.from_records(t_dicts)
    t_df.to_csv(data_dir / t_tests_csv, float_format='%.3f')
    return t_df


if __name__ == "__main__":
    human_knn_dir = similarity_plots_path / 'human_knn' / 'matches_20_restrict_future_True'
    run_t_test(human_knn_dir)
    bot_11_4 = similarity_plots_path / '_11_4_23_prebaked_no_mask_100_tries'
    run_t_test(bot_11_4)
    bot_11_2 = similarity_plots_path / '_11_2_23_prebaked_no_mask_100_tries'
    run_t_test(bot_11_2)


