from math import isqrt, ceil
from pathlib import Path

import torch
from einops import rearrange

from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, delta_pos_grid_radius, \
    delta_pos_grid_cell_dim, delta_pos_z_num_cells, delta_pos_grid_num_cells, delta_pos_grid_num_cells_per_xy_dim, \
    delta_pos_grid_num_xy_cells_per_z_change, num_radial_bins, num_radial_bins_per_z_axis, direction_angle_range, \
    StatureOptions
from learn_bot.libs.hdf5_to_pd import load_hdf5_extra_to_list, load_hdf5_to_pd
from learn_bot.libs.vec import Vec3
from dataclasses import dataclass
from typing import List, Tuple
from learn_bot.libs.io_transforms import CUDA_DEVICE_STR, CPU_DEVICE_STR

@dataclass
class AABB:
    min: Vec3
    max: Vec3


def nav_region_to_aabb(nav_region: List[int]) -> AABB:
    return AABB(Vec3(nav_region[0], nav_region[1], nav_region[2]), Vec3(nav_region[3], nav_region[4], nav_region[5]))


nav_above_below_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'nav' / 'de_dust2_nav_above_below.hdf5'


class NavData:
    nav_region: AABB
    num_steps: torch.Tensor
    nav_grid_base: torch.Tensor
    nav_above_below: torch.Tensor

    def __init__(self, device_str):
        self.nav_region = nav_region_to_aabb(load_hdf5_extra_to_list(nav_above_below_hdf5_data_path)[0])
        nav_above_below_df = load_hdf5_to_pd(nav_above_below_hdf5_data_path)
        self.nav_above_below = torch.Tensor(nav_above_below_df[['z nearest', 'z below']].to_numpy()).to(device_str)

        num_y_steps = int(ceil((self.nav_region.max.y - self.nav_region.min.y) / nav_step_size))
        num_z_steps = int(ceil((self.nav_region.max.z - self.nav_region.min.z) / nav_step_size))
        self.num_steps = torch.tensor([[[num_y_steps * num_z_steps, num_z_steps, 1]]]).to(device_str).long()
        self.nav_grid_base = torch.tensor([[[self.nav_region.min.x, self.nav_region.min.y, self.nav_region.min.z]]]).to(device_str)


# 130 units per half second (rounded up), 12.8 ticks per second
max_speed_per_half_second = 130.
max_speed_per_second = 250.
max_run_speed_per_tick = 130. / (12.8 / 2)
max_jump_height = 65.
nav_step_size = 10.


@dataclass
class DeltaXYZIndices:
    x_index: torch.Tensor
    y_index: torch.Tensor
    z_jump_index: torch.Tensor


def get_delta_indices_from_grid(pred_labels: torch.Tensor) -> DeltaXYZIndices:
    z_jump_index = torch.floor(pred_labels / delta_pos_grid_num_xy_cells_per_z_change)
    xy_pred_label = torch.floor(torch.remainder(pred_labels, delta_pos_grid_num_xy_cells_per_z_change))
    x_index = torch.floor(torch.remainder(xy_pred_label, delta_pos_grid_num_cells_per_xy_dim)) - \
              int(delta_pos_grid_num_cells_per_xy_dim / 2)
    y_index = torch.floor(xy_pred_label / delta_pos_grid_num_cells_per_xy_dim) - \
              int(delta_pos_grid_num_cells_per_xy_dim / 2)
    return DeltaXYZIndices(x_index, y_index, z_jump_index)


def get_delta_indices_from_radial(pred_labels: torch.Tensor, stature_to_speed: torch.Tensor) -> DeltaXYZIndices:
    not_moving = pred_labels == 0.
    moving_pred_labels = pred_labels - 1.
    z_jump_index = torch.floor(moving_pred_labels / num_radial_bins_per_z_axis)
    dir_stature_pred_label = torch.floor(moving_pred_labels / StatureOptions.NUM_STATURE_OPTIONS.value)
    per_batch_stature_index = \
        torch.floor(torch.remainder(dir_stature_pred_label, StatureOptions.NUM_STATURE_OPTIONS.value)).int()
    # necessary since index select expects 1d
    flattened_stature_index = rearrange(per_batch_stature_index, 'b p -> (b p)',
                                        p=len(specific_player_place_area_columns))
    dir_degrees = torch.floor(dir_stature_pred_label / delta_pos_grid_num_cells_per_xy_dim) * direction_angle_range
    # stature to lookup table of speeds
    flattened_max_speed_per_stature = torch.index_select(stature_to_speed, 0, flattened_stature_index)
    per_batch_max_speed_per_stature = rearrange(flattened_max_speed_per_stature, '(b p) -> b p',
                                                p=len(specific_player_place_area_columns))
    # return cos/sin of dir degre
    not_moving_zeros = torch.zeros_like(per_batch_max_speed_per_stature)
    return DeltaXYZIndices(
        torch.where(not_moving, not_moving_zeros, torch.cos(dir_degrees) * per_batch_max_speed_per_stature),
        torch.where(not_moving, not_moving_zeros, torch.sin(dir_degrees) * per_batch_max_speed_per_stature),
        torch.where(not_moving, not_moving_zeros, z_jump_index))


def compute_new_pos(input_pos_tensor: torch.Tensor, pred_labels: torch.Tensor, nav_data: NavData, pred_is_grid: bool,
                    stature_to_speed: torch.Tensor):
    delta_xyz_indices = get_delta_indices_from_grid(pred_labels) if pred_is_grid else \
        get_delta_indices_from_radial(pred_labels, stature_to_speed)
    z_index = torch.zeros_like(delta_xyz_indices.x_index)

    # convert to xy pos changes
    pos_index = torch.stack([delta_xyz_indices.x_index, delta_xyz_indices.y_index, z_index], dim=-1)
    unscaled_pos_change = (pos_index * delta_pos_grid_cell_dim)
    pos_change_norm = torch.linalg.vector_norm(unscaled_pos_change, dim=-1, keepdim=True)
    all_scaled_pos_change = (max_run_speed_per_tick / pos_change_norm) * unscaled_pos_change
    # if norm less than max run speed, don't scale
    scaled_pos_change = torch.where(pos_change_norm > max_run_speed_per_tick, all_scaled_pos_change,
                                    unscaled_pos_change)

    # apply to input pos
    output_pos_tensor = input_pos_tensor[:, :, 0, :] + scaled_pos_change
    output_pos_tensor[:, :, 2] += torch.where(delta_xyz_indices.z_jump_index == 2., max_jump_height, 0.)
    output_pos_tensor[:, :, 0] = output_pos_tensor[:, :, 0].clamp(min=nav_data.nav_region.min.x,
                                                                  max=nav_data.nav_region.max.x)
    output_pos_tensor[:, :, 1] = output_pos_tensor[:, :, 1].clamp(min=nav_data.nav_region.min.y,
                                                                  max=nav_data.nav_region.max.y)
    output_pos_tensor[:, :, 2] = output_pos_tensor[:, :, 2].clamp(min=nav_data.nav_region.min.z,
                                                                  max=nav_data.nav_region.max.z)

    # compute z pos
    nav_grid_steps = torch.floor((output_pos_tensor - nav_data.nav_grid_base) / nav_step_size).long()
    nav_grid_index = rearrange(torch.sum(nav_data.num_steps * nav_grid_steps, dim=-1), 'b p -> (b p)')
    nav_above_below_per_player = rearrange(torch.index_select(nav_data.nav_above_below, 0, nav_grid_index),
                                           '(b p) o -> b p o', p=len(specific_player_place_area_columns))
    output_pos_tensor[:, :, 2] = torch.where(delta_xyz_indices.z_jump_index == 1,
                                             nav_above_below_per_player[:, :, 0], nav_above_below_per_player[:, :, 1])
    output_and_history_pos_tensor = torch.roll(input_pos_tensor, 1, 2)
    output_and_history_pos_tensor[:, :, 0, :] = output_pos_tensor
    return output_and_history_pos_tensor


def one_hot_max_to_index(pred: torch.Tensor) -> torch.Tensor:
    return torch.argmax(rearrange(pred, 'b (p d) -> b p d', p=len(specific_player_place_area_columns)), 2)


def one_hot_prob_to_index(pred: torch.Tensor) -> torch.Tensor:
    if pred.device.type == CUDA_DEVICE_STR:
        probs = rearrange(pred, 'b (p d) -> (b p) d', p=len(specific_player_place_area_columns))
        return rearrange(torch.multinomial(probs, 1, replacement=True), '(b p) d -> b (p d)',
                         p=len(specific_player_place_area_columns))
    else:
        return one_hot_max_to_index(pred)
