import pickle
from dataclasses import dataclass
import time
from enum import Enum
from typing import Dict

import h5py
import numpy as np
from pathlib import Path

import pandas as pd

from learn_bot.latent.analyze.create_test_plant_states import test_plant_states_file_name, load_data_options
from learn_bot.latent.analyze.humanness_metrics.column_names import *
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.place_area.column_names import get_similarity_column
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.train import train_test_split_file_name
from learn_bot.libs.df_grouping import TrainTestSplit
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.multi_hdf5_wrapper import train_test_split_folder_path, make_train_test_splits_relative, \
    absolute_to_relative_train_test_key


class HumannessDataOptions(Enum):
    ROLLOUT = 1
    ALL_TRAIN = 2
    HEURISTIC = 3
    DEFAULT = 4


class HumannessMetrics:
    unscaled_speed: np.ndarray
    unscaled_speed_when_firing: np.ndarray
    unscaled_speed_when_shot: np.ndarray

    scaled_speed: np.ndarray
    scaled_speed_when_firing: np.ndarray
    scaled_speed_when_shot: np.ndarray

    weapon_only_scaled_speed: np.ndarray
    weapon_only_scaled_speed_when_firing: np.ndarray
    weapon_only_scaled_speed_when_shot: np.ndarray

    distance_to_nearest_teammate: np.ndarray
    distance_to_nearest_teammate_when_firing: np.ndarray
    distance_to_nearest_teammate_when_shot: np.ndarray

    distance_to_nearest_enemy: np.ndarray
    distance_to_nearest_enemy_when_firing: np.ndarray
    distance_to_nearest_enemy_when_shot: np.ndarray

    distance_to_attacker_when_shot: np.ndarray

    distance_to_cover: np.ndarray
    distance_to_cover_when_enemy_visible_no_fov: np.ndarray
    distance_to_cover_when_enemy_visible_fov: np.ndarray
    distance_to_cover_when_firing: np.ndarray
    distance_to_cover_when_shot: np.ndarray

    time_from_firing_to_teammate_seeing_enemy_fov: np.ndarray
    time_from_shot_to_teammate_seeing_enemy_fov: np.ndarray

    pct_time_max_speed_ct: np.ndarray
    pct_time_max_speed_t: np.ndarray
    pct_time_still_ct: np.ndarray
    pct_time_still_t: np.ndarray
    ct_wins: np.ndarray

    def __init__(self, data_option: HumannessDataOptions, limit_to_test: bool) -> np.ndarray:
        test_plant_states_df = load_hdf5_to_pd(train_test_split_folder_path / test_plant_states_file_name)

        # get data as numpy arrays and column names
        hdf5_paths: List[Path]
        if data_option == HumannessDataOptions.ROLLOUT:
            hdf5_paths = [rollout_humanness_hdf5_data_path]
        elif data_option == HumannessDataOptions.ALL_TRAIN:
            hdf5_paths = all_train_humanness_hdf5_data_paths
        elif data_option == HumannessDataOptions.HEURISTIC:
            hdf5_paths = [heuristic_humanness_hdf5_data_path]
        else:
            hdf5_paths = [default_humanness_hdf5_data_path]

        hdf5_to_test_round_ids: Dict[Path, List[int]] = {}
        if limit_to_test:
            load_data_result = LoadDataResult(load_data_options)
            for hdf5_wrapper in load_data_result.multi_hdf5_wrapper.hdf5_wrappers:
                round_df = hdf5_wrapper.id_df.groupby(round_id_column, as_index=False).first()
                push_round_df = round_df[round_df[get_similarity_column(0)]]
                hdf5_to_test_round_ids[absolute_to_relative_train_test_key(hdf5_wrapper.hdf5_path)] = \
                    list(push_round_df[round_id_column])

        first_file: bool = True
        for hdf5_path in hdf5_paths:
            splits_key_path = absolute_to_relative_train_test_key(Path(
                str(hdf5_path).replace('humannessMetrics', 'behaviorTreeTeamFeatureStore')))
            with h5py.File(hdf5_path) as hdf5_file:
                hdf5_data = hdf5_file['data']
                # actual data
                unscaled_speed = hdf5_data[unscaled_speed_name][...]
                unscaled_speed_when_firing = hdf5_data[unscaled_speed_when_firing_name][...]
                unscaled_speed_when_shot = hdf5_data[unscaled_speed_when_shot_name][...]

                scaled_speed = hdf5_data[scaled_speed_name][...]
                scaled_speed_when_firing = hdf5_data[scaled_speed_when_firing_name][...]
                scaled_speed_when_shot = hdf5_data[scaled_speed_when_shot_name][...]

                weapon_only_scaled_speed = hdf5_data[weapon_only_scaled_speed_name][...]
                weapon_only_scaled_speed_when_firing = hdf5_data[weapon_only_scaled_speed_when_firing_name][...]
                weapon_only_scaled_speed_when_shot = hdf5_data[weapon_only_scaled_speed_when_shot_name][...]

                distance_to_nearest_teammate = hdf5_data[distance_to_nearest_teammate_name][...]
                distance_to_nearest_teammate_when_firing = \
                    hdf5_data[distance_to_nearest_teammate_when_firing_name][...]
                distance_to_nearest_teammate_when_shot = hdf5_data[distance_to_nearest_teammate_when_shot_name][...]

                distance_to_nearest_enemy = hdf5_data[distance_to_nearest_enemy_name][...]
                distance_to_nearest_enemy_when_firing = hdf5_data[distance_to_nearest_enemy_when_firing_name][...]
                distance_to_nearest_enemy_when_shot = hdf5_data[distance_to_nearest_enemy_when_shot_name][...]

                distance_to_attacker_when_shot = hdf5_data[distance_to_attacker_when_shot_name][...]

                distance_to_cover = hdf5_data[distance_to_cover_name][...]
                distance_to_cover_when_enemy_visible_no_fov = \
                    hdf5_data[distance_to_cover_when_enemy_visible_no_fov_name][...]
                distance_to_cover_when_enemy_visible_fov = \
                    hdf5_data[distance_to_cover_when_enemy_visible_fov_name][...]
                distance_to_cover_when_firing = hdf5_data[distance_to_cover_when_firing_name][...]
                distance_to_cover_when_shot = hdf5_data[distance_to_cover_when_shot_name][...]

                time_from_firing_to_teammate_seeing_enemy_fov = \
                    hdf5_data[time_from_firing_to_teammate_seeing_enemy_fov_name][...]
                time_from_shot_to_teammate_seeing_enemy_fov = \
                    hdf5_data[time_from_shot_to_teammate_seeing_enemy_fov_name][...]

                pct_time_max_speed_ct = hdf5_data[pct_time_max_speed_ct_name][...]
                pct_time_max_speed_t = hdf5_data[pct_time_max_speed_t_name][...]
                pct_time_still_ct = hdf5_data[pct_time_still_ct_name][...]
                pct_time_still_t = hdf5_data[pct_time_still_t_name][...]
                ct_wins = hdf5_data[ct_wins_name][...]

                # round ids for filtering
                round_id_per_pat = hdf5_data[round_id_per_pat_name][...]
                round_id_per_firing_pat = hdf5_data[round_id_per_firing_pat_name][...]
                round_id_per_shot_pat = hdf5_data[round_id_per_shot_pat_name][...]
                round_id_per_nearest_teammate = hdf5_data[round_id_per_nearest_teammate_name][...]
                round_id_per_nearest_teammate_firing = hdf5_data[round_id_per_nearest_teammate_firing_name][...]
                round_id_per_nearest_teammate_shot = hdf5_data[round_id_per_nearest_teammate_shot_name][...]
                round_id_per_enemy_visible_no_fov_pat = hdf5_data[round_id_per_enemy_visible_no_fov_pat_name][...]
                round_id_per_enemy_visible_fov_pat = hdf5_data[round_id_per_enemy_visible_fov_pat_name][...]
                round_id_per_firing_to_teammate_seeing_enemy = \
                    hdf5_data[round_id_per_firing_to_teammate_seeing_enemy_name][...]
                round_id_per_shot_to_teammate_seeing_enemy = \
                    hdf5_data[round_id_per_shot_to_teammate_seeing_enemy_name][...]
                round_id_per_round = hdf5_data[round_id_per_round_name][...]

                # filter if necessary
                if limit_to_test:
                    test_round_ids = hdf5_to_test_round_ids[Path(splits_key_path)]
                    test_round_id_per_pat = np.isin(round_id_per_pat, test_round_ids)
                    test_round_id_per_firing_pat = np.isin(round_id_per_firing_pat, test_round_ids)
                    test_round_id_per_shot_pat = np.isin(round_id_per_shot_pat, test_round_ids)
                    test_round_id_per_nearest_teammate = \
                        np.isin(round_id_per_nearest_teammate, test_round_ids)
                    test_round_id_per_nearest_teammate_firing = \
                        np.isin(round_id_per_nearest_teammate_firing, test_round_ids)
                    test_round_id_per_nearest_teammate_shot = \
                        np.isin(round_id_per_nearest_teammate_shot, test_round_ids)
                    test_round_id_per_enemy_visible_no_fov_pat = \
                        np.isin(round_id_per_enemy_visible_no_fov_pat, test_round_ids)
                    test_round_id_per_enemy_visible_fov_pat = \
                        np.isin(round_id_per_enemy_visible_fov_pat, test_round_ids)
                    test_round_id_per_firing_to_teammate_seeing_enemy = \
                        np.isin(round_id_per_firing_to_teammate_seeing_enemy, test_round_ids)
                    test_round_id_per_shot_to_teammate_seeing_enemy = \
                        np.isin(round_id_per_shot_to_teammate_seeing_enemy, test_round_ids)
                    test_round_id_per_round = np.isin(round_id_per_round, test_round_ids)

                    unscaled_speed = unscaled_speed[test_round_id_per_pat]
                    unscaled_speed_when_firing = unscaled_speed_when_firing[test_round_id_per_firing_pat]
                    unscaled_speed_when_shot = unscaled_speed_when_shot[test_round_id_per_shot_pat]

                    scaled_speed = scaled_speed[test_round_id_per_pat]
                    scaled_speed_when_firing = scaled_speed_when_firing[test_round_id_per_firing_pat]
                    scaled_speed_when_shot = scaled_speed_when_shot[test_round_id_per_shot_pat]

                    weapon_only_scaled_speed = weapon_only_scaled_speed[test_round_id_per_pat]
                    weapon_only_scaled_speed_when_firing = \
                        weapon_only_scaled_speed_when_firing[test_round_id_per_firing_pat]
                    weapon_only_scaled_speed_when_shot = \
                        weapon_only_scaled_speed_when_shot[test_round_id_per_shot_pat]

                    distance_to_nearest_teammate = distance_to_nearest_teammate[test_round_id_per_nearest_teammate]
                    distance_to_nearest_teammate_when_firing = \
                        distance_to_nearest_teammate_when_firing[test_round_id_per_nearest_teammate_firing]
                    distance_to_nearest_teammate_when_shot = \
                        distance_to_nearest_teammate_when_shot[test_round_id_per_nearest_teammate_shot]

                    distance_to_nearest_enemy = distance_to_nearest_enemy[test_round_id_per_pat]
                    distance_to_nearest_enemy_when_firing = \
                        distance_to_nearest_enemy_when_firing[test_round_id_per_firing_pat]
                    distance_to_nearest_enemy_when_shot = \
                        distance_to_nearest_enemy_when_shot[test_round_id_per_shot_pat]

                    distance_to_attacker_when_shot = \
                        distance_to_attacker_when_shot[test_round_id_per_shot_pat]

                    distance_to_cover = distance_to_cover[test_round_id_per_pat]
                    distance_to_cover_when_enemy_visible_no_fov = \
                        distance_to_cover_when_enemy_visible_no_fov[test_round_id_per_enemy_visible_no_fov_pat]
                    distance_to_cover_when_enemy_visible_fov = \
                        distance_to_cover_when_enemy_visible_fov[test_round_id_per_enemy_visible_fov_pat]
                    distance_to_cover_when_firing = distance_to_cover_when_firing[test_round_id_per_firing_pat]
                    distance_to_cover_when_shot = distance_to_cover_when_shot[test_round_id_per_shot_pat]

                    time_from_firing_to_teammate_seeing_enemy_fov = \
                        time_from_firing_to_teammate_seeing_enemy_fov[test_round_id_per_firing_to_teammate_seeing_enemy]
                    time_from_shot_to_teammate_seeing_enemy_fov = \
                        time_from_shot_to_teammate_seeing_enemy_fov[test_round_id_per_shot_to_teammate_seeing_enemy]

                    pct_time_max_speed_ct = pct_time_max_speed_ct[test_round_id_per_round]
                    pct_time_max_speed_t = pct_time_max_speed_t[test_round_id_per_round]
                    pct_time_still_ct = pct_time_still_ct[test_round_id_per_round]
                    pct_time_still_t = pct_time_still_t[test_round_id_per_round]
                    ct_wins = ct_wins[test_round_id_per_round]

                if first_file:
                    self.unscaled_speed = unscaled_speed
                    self.unscaled_speed_when_firing = unscaled_speed_when_firing
                    self.unscaled_speed_when_shot = unscaled_speed_when_shot

                    self.scaled_speed = scaled_speed
                    self.scaled_speed_when_firing = scaled_speed_when_firing
                    self.scaled_speed_when_shot = scaled_speed_when_shot

                    self.weapon_only_scaled_speed = weapon_only_scaled_speed
                    self.weapon_only_scaled_speed_when_firing = weapon_only_scaled_speed_when_firing
                    self.weapon_only_scaled_speed_when_shot = weapon_only_scaled_speed_when_shot

                    self.distance_to_nearest_teammate = distance_to_nearest_teammate
                    self.distance_to_nearest_teammate_when_firing = distance_to_nearest_teammate_when_firing
                    self.distance_to_nearest_teammate_when_shot = distance_to_nearest_teammate_when_shot

                    self.distance_to_nearest_enemy = distance_to_nearest_enemy
                    self.distance_to_nearest_enemy_when_firing = distance_to_nearest_enemy_when_firing
                    self.distance_to_nearest_enemy_when_shot = distance_to_nearest_enemy_when_shot

                    self.distance_to_attacker_when_shot = distance_to_attacker_when_shot

                    self.distance_to_cover = distance_to_cover
                    self.distance_to_cover_when_enemy_visible_no_fov = distance_to_cover_when_enemy_visible_no_fov
                    self.distance_to_cover_when_enemy_visible_fov = distance_to_cover_when_enemy_visible_fov
                    self.distance_to_cover_when_firing = distance_to_cover_when_firing
                    self.distance_to_cover_when_shot = distance_to_cover_when_shot

                    self.time_from_firing_to_teammate_seeing_enemy_fov = time_from_firing_to_teammate_seeing_enemy_fov
                    self.time_from_shot_to_teammate_seeing_enemy_fov = time_from_shot_to_teammate_seeing_enemy_fov

                    self.pct_time_max_speed_ct = pct_time_max_speed_ct
                    self.pct_time_max_speed_t = pct_time_max_speed_t
                    self.pct_time_still_ct = pct_time_still_ct
                    self.pct_time_still_t = pct_time_still_t
                    self.ct_wins = ct_wins
                else:
                    self.unscaled_speed = np.append(self.unscaled_speed, unscaled_speed)
                    self.unscaled_speed_when_firing = \
                        np.append(self.unscaled_speed_when_firing, unscaled_speed_when_firing)
                    self.unscaled_speed_when_shot = \
                        np.append(self.unscaled_speed_when_shot, unscaled_speed_when_shot)

                    self.scaled_speed = np.append(self.scaled_speed, scaled_speed)
                    self.scaled_speed_when_firing = \
                        np.append(self.scaled_speed_when_firing, scaled_speed_when_firing)
                    self.scaled_speed_when_shot = \
                        np.append(self.scaled_speed_when_shot, scaled_speed_when_shot)

                    self.weapon_only_scaled_speed = \
                        np.append(self.weapon_only_scaled_speed, weapon_only_scaled_speed)
                    self.weapon_only_scaled_speed_when_firing = \
                        np.append(self.weapon_only_scaled_speed_when_firing, weapon_only_scaled_speed_when_firing)
                    self.weapon_only_scaled_speed_when_shot = \
                        np.append(self.weapon_only_scaled_speed_when_shot, weapon_only_scaled_speed_when_shot)

                    self.distance_to_nearest_teammate = \
                        np.append(self.distance_to_nearest_teammate, distance_to_nearest_teammate)
                    self.distance_to_nearest_teammate_when_firing = \
                        np.append(self.distance_to_nearest_teammate_when_firing,
                                  distance_to_nearest_teammate_when_firing)
                    self.distance_to_nearest_teammate_when_shot = \
                        np.append(self.distance_to_nearest_teammate_when_shot, distance_to_nearest_teammate_when_shot)

                    self.distance_to_nearest_enemy = \
                        np.append(self.distance_to_nearest_enemy, distance_to_nearest_enemy)
                    self.distance_to_nearest_enemy_when_firing = \
                        np.append(self.distance_to_nearest_enemy_when_firing, distance_to_nearest_enemy_when_firing)
                    self.distance_to_nearest_enemy_when_shot = \
                        np.append(self.distance_to_nearest_enemy_when_shot, distance_to_nearest_enemy_when_shot)

                    self.distance_to_attacker_when_shot = \
                        np.append(self.distance_to_attacker_when_shot, distance_to_attacker_when_shot)

                    self.distance_to_cover = \
                        np.append(self.distance_to_cover, distance_to_cover)
                    self.distance_to_cover_when_enemy_visible_no_fov = \
                        np.append(self.distance_to_cover_when_enemy_visible_no_fov,
                                  distance_to_cover_when_enemy_visible_no_fov)
                    self.distance_to_cover_when_enemy_visible_fov = \
                        np.append(self.distance_to_cover_when_enemy_visible_fov,
                                  distance_to_cover_when_enemy_visible_fov)
                    self.distance_to_cover_when_firing = \
                        np.append(self.distance_to_cover_when_firing, distance_to_cover_when_firing)
                    self.distance_to_cover_when_shot = \
                        np.append(self.distance_to_cover_when_shot, distance_to_cover_when_shot)

                    self.time_from_firing_to_teammate_seeing_enemy_fov = \
                        np.append(self.time_from_firing_to_teammate_seeing_enemy_fov,
                                  time_from_firing_to_teammate_seeing_enemy_fov)
                    self.time_from_shot_to_teammate_seeing_enemy_fov = \
                        np.append(self.time_from_shot_to_teammate_seeing_enemy_fov,
                                  time_from_shot_to_teammate_seeing_enemy_fov)

                    self.pct_time_max_speed_ct = np.append(self.pct_time_max_speed_ct, pct_time_max_speed_ct)
                    self.pct_time_max_speed_t = np.append(self.pct_time_max_speed_t, pct_time_max_speed_t)
                    self.pct_time_still_ct = np.append(self.pct_time_still_ct, pct_time_still_ct)
                    self.pct_time_still_t = np.append(self.pct_time_still_t, pct_time_still_t)
                    self.ct_wins = np.append(self.ct_wins, ct_wins)

                first_file = False


