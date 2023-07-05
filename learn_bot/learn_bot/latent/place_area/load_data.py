from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable

import pandas as pd

from learn_bot.latent.engagement.column_names import game_id_column, round_id_column
from learn_bot.latent.place_area.column_names import hdf5_id_columns, test_success_col
from learn_bot.latent.place_area.create_test_data import create_left_right_train_data, create_left_right_test_data
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper, PDWrapper
from learn_bot.libs.multi_hdf5_wrapper import MultiHDF5Wrapper, HDF5SourceOptions

just_test_comment = "just_test"
just_bot_comment = "just_bot"
bot_and_human_comment = "bot_and_human"
just_small_human_comment = "just_small_human"
just_human_comment = "just_human"
all_comment = "_all"
human_with_bot_nav_added_comment = "human_with_added_bot_nav"
limited_comment = "_limited"
curriculum_comment = "_curriculum"

all_train_latent_team_hdf5_dir_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs'
human_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'
small_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'smallBehaviorTreeTeamFeatureStore.parquet'
manual_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'manual_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'
rollout_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'

LimitFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class LoadDataOptions:
    use_manual_data: bool
    use_rollout_data: bool
    use_synthetic_data: bool
    use_small_human_data: bool
    use_all_human_data: bool
    add_manual_to_all_human_data: bool
    limit_manual_data_to_no_enemies_nav: bool


class LoadDataResult:
    diff_train_test: bool
    dataset_comment: str
    multi_hdf5_wrapper: MultiHDF5Wrapper

    def __init__(self, load_data_options: LoadDataOptions):
        self.diff_train_test = True
        force_test_data = None
        hdf5_sources: List[HDF5SourceOptions] = []
        duplicate_last_hdf5_equal_to_rest = False
        if load_data_options.use_manual_data:
            self.dataset_comment = just_bot_comment
            manual_data = HDF5Wrapper(manual_latent_team_hdf5_data_path, hdf5_id_columns)
            if load_data_options.limit_manual_data_to_no_enemies_nav:
                manual_data.limit((manual_data.id_df[test_success_col] == 1.) & (manual_data.id_df[game_id_column] == 1))
            else:
                manual_data.limit(manual_data.id_df[test_success_col] == 1.)
            hdf5_sources.append(manual_data)
        elif load_data_options.use_rollout_data:
            self.dataset_comment = "rollout"
            rollout_data = HDF5Wrapper(rollout_latent_team_hdf5_data_path, hdf5_id_columns)
            self.diff_train_test = False
            hdf5_sources.append(rollout_data)
        elif load_data_options.use_synthetic_data:
            self.dataset_comment = just_test_comment
            base_data = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=[i for i in range(1)])
            synthetic_data_df = create_left_right_train_data(base_data)
            synthetic_data = PDWrapper('train', synthetic_data_df, hdf5_id_columns)
            force_test_data_df = create_left_right_test_data(base_data)
            force_test_data = PDWrapper('test', force_test_data_df, hdf5_id_columns)
            self.diff_train_test = False
            hdf5_sources.append(synthetic_data)
        elif load_data_options.use_small_human_data:
            self.dataset_comment = just_human_comment + limited_comment + "_unfilitered"
            human_data = HDF5Wrapper(human_latent_team_hdf5_data_path, ['id', round_id_column, test_success_col])
            #with open(good_retake_rounds_path, "r") as f:
            #    good_retake_rounds = eval(f.read())
            #human_data.limit(human_data.id_df[round_id_column].isin(good_retake_rounds))
            hdf5_sources.append(human_data)
        elif load_data_options.use_all_human_data:
            self.dataset_comment = just_human_comment + all_comment
            hdf5_sources.append(all_train_latent_team_hdf5_dir_path)
            # NEED WAY TO RESTRICT TO GOOD ROUNDS
            if load_data_options.add_manual_to_all_human_data:
                self.dataset_comment = human_with_bot_nav_added_comment
                manual_data = HDF5Wrapper(manual_latent_team_hdf5_data_path, hdf5_id_columns)
                if load_data_options.limit_manual_data_to_no_enemies_nav:
                    manual_data.limit((manual_data.id_df[test_success_col] == 1.) & (manual_data.id_df[game_id_column] == 1))
                else:
                    manual_data.limit(manual_data.id_df[test_success_col] == 1.)
                hdf5_sources.append(manual_data)
                duplicate_last_hdf5_equal_to_rest = True
        else:
            raise Exception("Must call LoadDataResult with something True")
        self.multi_hdf5_wrapper = MultiHDF5Wrapper(hdf5_sources, hdf5_id_columns, diff_train_test=self.diff_train_test,
                                                   force_test_hdf5=force_test_data,
                                                   duplicate_last_hdf5_equal_to_rest=duplicate_last_hdf5_equal_to_rest)

    def limit(self, limit_fns: List[LimitFn]):
        for i in range(len(limit_fns)):
            self.multi_hdf5_wrapper.hdf5_wrappers[i].limit(limit_fns[i](self.multi_hdf5_wrapper.hdf5_wrappers[i].id_df))




