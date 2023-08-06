from dataclasses import dataclass
from enum import Enum

import h5py
import numpy as np
from pathlib import Path
from learn_bot.latent.analyze.humanness_metrics.column_names import *


class HumannessDataOptions(Enum):
    ROLLOUT = 1
    ALL_TRAIN = 2


class HumannessMetrics:
    velocity_when_firing: np.ndarray
    velocity_when_shot: np.ndarray
    distance_to_nearest_teammate_when_firing: np.ndarray
    distance_to_nearest_enemy_when_firing: np.ndarray
    distance_to_nearest_teammate_when_shot: np.ndarray
    distance_to_attacker_when_shot: np.ndarray
    distance_to_cover_when_firing: np.ndarray
    distance_to_cover_when_shot: np.ndarray
    pct_time_max_speed_ct: np.ndarray
    pct_time_max_speed_t: np.ndarray
    pct_time_still_ct: np.ndarray
    pct_time_still_t: np.ndarray
    ct_wins: np.ndarray

    def __init__(self, data_option: HumannessDataOptions) -> np.ndarray:
        # get data as numpy arrays and column names
        hdf5_paths: List[Path]
        if data_option == HumannessDataOptions.ROLLOUT:
            hdf5_paths = [rollout_humanness_hdf5_data_path]
        else:
            hdf5_paths = all_train_humanness_hdf5_data_paths

        first_file: bool = True
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path) as hdf5_file:
                hdf5_data = hdf5_file['data']

                if first_file:
                    self.velocity_when_firing = hdf5_data[velocity_when_firing_name][...]
                    self.velocity_when_shot = hdf5_data[velocity_when_shot_name][...]
                    self.distance_to_nearest_teammate_when_firing = hdf5_data[distance_to_nearest_teammate_when_firing_name][...]
                    self.distance_to_nearest_enemy_when_firing = hdf5_data[distance_to_nearest_enemy_when_firing_name][...]
                    self.distance_to_nearest_teammate_when_shot = hdf5_data[distance_to_nearest_teammate_when_shot_name][...]
                    self.distance_to_attacker_when_shot = hdf5_data[distance_to_attacker_when_shot_name][...]
                    self.distance_to_cover_when_firing = hdf5_data[distance_to_cover_when_firing_name][...]
                    self.distance_to_cover_when_shot = hdf5_data[distance_to_cover_when_shot_name][...]
                    self.pct_time_max_speed_ct = hdf5_data[pct_time_max_speed_ct_name][...]
                    self.pct_time_max_speed_t = hdf5_data[pct_time_max_speed_t_name][...]
                    self.pct_time_still_ct = hdf5_data[pct_time_still_ct_name][...]
                    self.pct_time_still_t = hdf5_data[pct_time_still_t_name][...]
                    self.ct_wins = hdf5_data[ct_wins_name][...]
                else:
                    self.velocity_when_firing = np.append(self.velocity_when_firing, hdf5_data[velocity_when_firing_name][...])
                    self.velocity_when_shot = np.append(self.velocity_when_shot, hdf5_data[velocity_when_shot_name][...])
                    self.distance_to_nearest_teammate_when_firing = \
                        np.append(self.distance_to_nearest_teammate_when_firing,
                                  hdf5_data[distance_to_nearest_teammate_when_firing_name][...])
                    self.distance_to_nearest_enemy_when_firing = \
                        np.append(self.distance_to_nearest_enemy_when_firing,
                                  hdf5_data[distance_to_nearest_enemy_when_firing_name][...])
                    self.distance_to_nearest_teammate_when_shot = \
                        np.append(self.distance_to_nearest_teammate_when_shot,
                                  hdf5_data[distance_to_nearest_teammate_when_shot_name][...],)
                    self.distance_to_attacker_when_shot = \
                        np.append(self.distance_to_attacker_when_shot,
                                  hdf5_data[distance_to_attacker_when_shot_name][...])
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


