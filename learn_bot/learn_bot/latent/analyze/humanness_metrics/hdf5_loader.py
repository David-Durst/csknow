import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Dict

import h5py
import numpy as np
from pathlib import Path

import pandas as pd

from learn_bot.latent.analyze.humanness_metrics.column_names import *
from learn_bot.latent.train import train_test_split_file_name
from learn_bot.libs.df_grouping import TrainTestSplit


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

    pct_time_max_speed_ct: np.ndarray
    pct_time_max_speed_t: np.ndarray
    pct_time_still_ct: np.ndarray
    pct_time_still_t: np.ndarray
    ct_wins: np.ndarray

    def __init__(self, data_option: HumannessDataOptions, limit_to_test: bool) -> np.ndarray:
        train_test_splits: Dict[Path, TrainTestSplit] = pickle.load(train_test_split_file_name)

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

        first_file: bool = True
        for hdf5_path in hdf5_paths:
            splits_key_path = str(hdf5_path).replace('humannessMetrics', 'behaviorTreeTeamFeatureStore')
            with h5py.File(hdf5_path) as hdf5_file:
                hdf5_data = hdf5_file['data']

                unscaled_speed = hdf5_data[unscaled_speed_name][...]
                unscaled_speed_when_firing = hdf5_data[unscaled_speed_when_firing_name][...]
                unscaled_speed_when_shot = hdf5_data[unscaled_speed_when_shot_name][...]

                scaled_speed = hdf5_data[scaled_speed_name][...]
                scaled_speed_when_firing = hdf5_data[scaled_speed_when_firing_name][...]
                scaled_speed_when_shot = hdf5_data[scaled_speed_when_shot_name][...]

                weapon_only_scaled_speed = hdf5_data[scaled_speed_name][...]
                weapon_only_scaled_speed_when_firing = hdf5_data[scaled_speed_when_firing_name][...]
                weapon_only_scaled_speed_when_shot = hdf5_data[scaled_speed_when_shot_name][...]

                distance_to_nearest_teammate = hdf5_data[distance_to_nearest_teammate_name][...]
                distance_to_nearest_teammate_when_firing = \
                    hdf5_data[distance_to_nearest_teammate_when_firing_name][...]
                distance_to_nearest_teammate_when_shot = hdf5_data[distance_to_nearest_teammate_when_shot_name][...]

                distance_to_nearest_enemy = hdf5_data[distance_to_nearest_enemy_when_firing_name][...]
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

                pct_time_max_speed_ct = hdf5_data[pct_time_max_speed_ct_name][...]
                pct_time_max_speed_t = hdf5_data[pct_time_max_speed_t_name][...]
                pct_time_still_ct = hdf5_data[pct_time_still_ct_name][...]
                pct_time_still_t = hdf5_data[pct_time_still_t_name][...]
                ct_wins = hdf5_data[ct_wins_name][...]

                # round id names
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

                # create dfs and filter them if necessary
                per_pat_df = pd.DataFrame([unscaled_speed, scaled_speed, weapon_only_scaled_speed,
                                           distance_to_nearest_enemy, distance_to_cover, round_id_per_pat])

                per_firing_pat_df = pd.DataFrame([unscaled_speed_when_firing, scaled_speed_when_firing,
                                                  weapon_only_scaled_speed_when_firing,
                                                  distance_to_nearest_enemy_when_firing, distance_to_cover_when_firing,
                                                  round_id_per_firing_pat])

                per_shot_pat_df = pd.DataFrame([unscaled_speed_when_shot, scaled_speed_when_shot,
                                                weapon_only_scaled_speed_when_shot,
                                                distance_to_nearest_enemy_when_shot, distance_to_attacker_when_shot,
                                                distance_to_cover_when_shot, round_id_per_shot_pat])

                nearest_teammate_pat_df = pd.DataFrame([distance_to_nearest_teammate, round_id_per_nearest_teammate])
                nearest_teammate_firing_pat_df = pd.DataFrame([distance_to_nearest_teammate_when_firing,
                                                               round_id_per_nearest_teammate_firing])
                nearest_teammate_shot_pat_df = pd.DataFrame([distance_to_nearest_teammate_when_shot,
                                                             round_id_per_nearest_teammate_shot])


                enemy_visible_no_fov_pat_df = pd.DataFrame([distance_to_cover_when_enemy_visible_no_fov,
                                                            round_id_per_enemy_visible_no_fov_pat])
                enemy_visible_fov_pat_df = pd.DataFrame([distance_to_cover_when_enemy_visible_fov,
                                                         round_id_per_enemy_visible_fov_pat])



                if first_file:
                    self.unscaled_speed = hdf5_data[unscaled_speed_name][...]
                    self.unscaled_speed_when_firing = hdf5_data[unscaled_speed_when_firing_name][...]
                    self.unscaled_speed_when_shot = hdf5_data[unscaled_speed_when_shot_name][...]

                    self.scaled_speed = hdf5_data[scaled_speed_name][...]
                    self.scaled_speed_when_firing = hdf5_data[scaled_speed_when_firing_name][...]
                    self.scaled_speed_when_shot = hdf5_data[scaled_speed_when_shot_name][...]

                    self.weapon_only_scaled_speed = hdf5_data[scaled_speed_name][...]
                    self.weapon_only_scaled_speed_when_firing = hdf5_data[scaled_speed_when_firing_name][...]
                    self.weapon_only_scaled_speed_when_shot = hdf5_data[scaled_speed_when_shot_name][...]

                    self.distance_to_nearest_teammate = hdf5_data[distance_to_nearest_teammate_name][...]
                    self.distance_to_nearest_teammate_when_firing = \
                        hdf5_data[distance_to_nearest_teammate_when_firing_name][...]
                    self.distance_to_nearest_teammate_when_shot = hdf5_data[distance_to_nearest_teammate_when_shot_name][...]

                    self.distance_to_nearest_enemy = hdf5_data[distance_to_nearest_enemy_when_firing_name][...]
                    self.distance_to_nearest_enemy_when_firing = hdf5_data[distance_to_nearest_enemy_when_firing_name][...]
                    self.distance_to_nearest_enemy_when_shot = hdf5_data[distance_to_nearest_enemy_when_shot_name][...]

                    self.distance_to_attacker_when_shot = hdf5_data[distance_to_attacker_when_shot_name][...]

                    self.distance_to_cover = hdf5_data[distance_to_cover_name][...]
                    self.distance_to_cover_when_enemy_visible_no_fov = \
                        hdf5_data[distance_to_cover_when_enemy_visible_no_fov_name][...]
                    self.distance_to_cover_when_enemy_visible_fov = \
                        hdf5_data[distance_to_cover_when_enemy_visible_fov_name][...]
                    self.distance_to_cover_when_firing = hdf5_data[distance_to_cover_when_firing_name][...]
                    self.distance_to_cover_when_shot = hdf5_data[distance_to_cover_when_shot_name][...]

                    self.pct_time_max_speed_ct = hdf5_data[pct_time_max_speed_ct_name][...]
                    self.pct_time_max_speed_t = hdf5_data[pct_time_max_speed_t_name][...]
                    self.pct_time_still_ct = hdf5_data[pct_time_still_ct_name][...]
                    self.pct_time_still_t = hdf5_data[pct_time_still_t_name][...]
                    self.ct_wins = hdf5_data[ct_wins_name][...]
                else:
                    self.unscaled_speed = np.append(self.unscaled_speed, hdf5_data[unscaled_speed_name][...])
                    self.unscaled_speed_when_firing = \
                        np.append(self.unscaled_speed_when_firing, hdf5_data[unscaled_speed_when_firing_name][...])
                    self.unscaled_speed_when_shot = \
                        np.append(self.unscaled_speed_when_shot, hdf5_data[unscaled_speed_when_shot_name][...])

                    self.scaled_speed = np.append(self.scaled_speed, hdf5_data[scaled_speed_name][...])
                    self.scaled_speed_when_firing = \
                        np.append(self.scaled_speed_when_firing, hdf5_data[scaled_speed_when_firing_name][...])
                    self.scaled_speed_when_shot = \
                        np.append(self.scaled_speed_when_shot, hdf5_data[scaled_speed_when_shot_name][...])

                    self.weapon_only_scaled_speed = \
                        np.append(self.weapon_only_scaled_speed, hdf5_data[weapon_only_scaled_speed_name][...])
                    self.weapon_only_scaled_speed_when_firing = \
                        np.append(self.weapon_only_scaled_speed_when_firing,
                                  hdf5_data[weapon_only_scaled_speed_when_firing_name][...])
                    self.weapon_only_scaled_speed_when_shot = \
                        np.append(self.weapon_only_scaled_speed_when_shot,
                                  hdf5_data[weapon_only_scaled_speed_when_shot_name][...])

                    self.distance_to_nearest_teammate = \
                        np.append(self.distance_to_nearest_teammate,
                                  hdf5_data[distance_to_nearest_teammate_name][...])
                    self.distance_to_nearest_teammate_when_firing = \
                        np.append(self.distance_to_nearest_teammate_when_firing,
                                  hdf5_data[distance_to_nearest_teammate_when_firing_name][...])
                    self.distance_to_nearest_teammate_when_shot = \
                        np.append(self.distance_to_nearest_teammate_when_shot,
                                  hdf5_data[distance_to_nearest_teammate_when_shot_name][...],)

                    self.distance_to_nearest_enemy = \
                        np.append(self.distance_to_nearest_enemy,
                                  hdf5_data[distance_to_nearest_enemy_name][...])
                    self.distance_to_nearest_enemy_when_firing = \
                        np.append(self.distance_to_nearest_enemy_when_firing,
                                  hdf5_data[distance_to_nearest_enemy_when_firing_name][...])
                    self.distance_to_nearest_enemy_when_shot = \
                        np.append(self.distance_to_nearest_enemy_when_shot,
                                  hdf5_data[distance_to_nearest_enemy_when_shot_name][...])

                    self.distance_to_attacker_when_shot = \
                        np.append(self.distance_to_attacker_when_shot,
                                  hdf5_data[distance_to_attacker_when_shot_name][...])

                    self.distance_to_cover = \
                        np.append(self.distance_to_cover, hdf5_data[distance_to_cover_name][...])
                    self.distance_to_cover_when_enemy_visible_no_fov = \
                        np.append(self.distance_to_cover_when_enemy_visible_no_fov,
                                  hdf5_data[distance_to_cover_when_enemy_visible_no_fov_name][...])
                    self.distance_to_cover_when_enemy_visible_fov = \
                        np.append(self.distance_to_cover_when_enemy_visible_fov,
                                  hdf5_data[distance_to_cover_when_enemy_visible_fov_name][...])
                    self.distance_to_cover_when_firing = \
                        np.append(self.distance_to_cover_when_firing,
                                  hdf5_data[distance_to_cover_when_firing_name][...])
                    self.distance_to_cover_when_shot = \
                        np.append(self.distance_to_cover_when_shot, hdf5_data[distance_to_cover_when_shot_name][...])

                    self.pct_time_max_speed_ct = \
                        np.append(self.pct_time_still_ct, hdf5_data[pct_time_max_speed_ct_name][...])
                    self.pct_time_max_speed_t = \
                        np.append(self.pct_time_still_t, hdf5_data[pct_time_max_speed_t_name][...])
                    self.pct_time_still_ct = np.append(self.pct_time_still_ct, hdf5_data[pct_time_still_ct_name][...])
                    self.pct_time_still_t = np.append(self.pct_time_still_t, hdf5_data[pct_time_still_t_name][...])
                    self.ct_wins = np.append(self.ct_wins, hdf5_data[ct_wins_name][...])

                first_file = False


