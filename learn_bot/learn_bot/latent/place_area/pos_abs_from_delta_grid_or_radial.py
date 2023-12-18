from math import isqrt, ceil
from pathlib import Path

import torch
from einops import rearrange, repeat

from learn_bot.latent.order.column_names import num_future_ticks, num_radial_ticks
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, delta_pos_grid_radius, \
    delta_pos_grid_cell_dim, delta_pos_z_num_cells, delta_pos_grid_num_cells, delta_pos_grid_num_cells_per_xy_dim, \
    delta_pos_grid_num_xy_cells_per_z_change, num_radial_bins, num_radial_bins_per_z_axis, direction_angle_range, \
    StatureOptions, vis_columns_names_to_index
from learn_bot.libs.hdf5_to_pd import load_hdf5_extra_to_list, load_hdf5_to_pd
from learn_bot.libs.vec import Vec3
from dataclasses import dataclass
from typing import List, Tuple, Optional
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
    nav_below_or_in: torch.Tensor

    def __init__(self, device_str):
        self.nav_region = nav_region_to_aabb(load_hdf5_extra_to_list(nav_above_below_hdf5_data_path)[0])
        nav_above_below_df = load_hdf5_to_pd(nav_above_below_hdf5_data_path)
        self.nav_below_or_in = torch.Tensor(nav_above_below_df['z below or in'].to_numpy()).to(device_str)

        num_y_steps = int(ceil((self.nav_region.max.y - self.nav_region.min.y) / nav_step_size))
        num_z_steps = int(ceil((self.nav_region.max.z - self.nav_region.min.z) / nav_step_size))
        # records number of steps in matrix per 1 step in this dimension: x is outer, so 1 steps in x is a full step in all y and z
        # then 1 step in y is all full step in z, and finally 1 step in z is just 1 step
        self.num_steps = torch.tensor([[[num_y_steps * num_z_steps, num_z_steps, 1]]]).to(device_str).long()
        self.nav_grid_base = torch.tensor([[[self.nav_region.min.x, self.nav_region.min.y, self.nav_region.min.z]]]).to(device_str)


# 130 units per half second (rounded up), 12.8 ticks per second
max_speed_per_second = 250.
# 130 is rounded up speed for half second, mul by 2 to sacle to full second, divide by 12.8 as 12.8 ticks per second
# (128 tick server, decimated to 1/10th rate in data set) and jumping by 6 ticks at a time to approximate half second of movement
data_ticks_per_sim_tick = 2
data_ticks_per_second = 16
max_run_speed_per_sim_tick = max_speed_per_second / float(data_ticks_per_second) * data_ticks_per_sim_tick
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


@dataclass
class DeltaPosWithZIndex:
    delta_pos: torch.Tensor
    z_jump_index: torch.Tensor


weapon_id_cols = [vis_columns_names_to_index[player_place_area_columns.player_weapon_id]
                  for player_place_area_columns in specific_player_place_area_columns]
scoped_cols = [vis_columns_names_to_index[player_place_area_columns.player_scoped]
                  for player_place_area_columns in specific_player_place_area_columns]


def get_delta_pos_from_radial(pred_labels: torch.Tensor, vis_tensor: Optional[torch.Tensor], stature_to_speed: torch.Tensor,
                              weapon_scoped_to_max_speed: Optional[torch.Tensor]) -> DeltaPosWithZIndex:
    not_moving = pred_labels == 0.
    moving_pred_labels = pred_labels - 1.
    not_moving_z_index = torch.zeros_like(moving_pred_labels)
    # if not moving, then z jump index is 0
    z_jump_index = torch.where(not_moving, not_moving_z_index,
                               torch.floor(moving_pred_labels / num_radial_bins_per_z_axis))
    dir_stature_pred_label = torch.floor(torch.remainder(moving_pred_labels, num_radial_bins_per_z_axis))
    per_batch_stature_index = \
        torch.floor(torch.remainder(dir_stature_pred_label, StatureOptions.NUM_STATURE_OPTIONS.value)).int()
    # necessary since index select expects 1d
    player_time_steps = len(specific_player_place_area_columns) * num_radial_ticks
    flattened_stature_index = rearrange(per_batch_stature_index, 'b pt -> (b pt)',
                                        pt=player_time_steps)
    dir_degrees = torch.floor(dir_stature_pred_label / StatureOptions.NUM_STATURE_OPTIONS.value) * direction_angle_range
    # stature to lookup table of speeds
    flattened_max_speed_per_stature = torch.index_select(stature_to_speed, 0, flattened_stature_index)
    # since pytorch doesn't support nested index_select, flatten it and I'll do lookup index computation my self
    if vis_tensor is not None:
        flattened_weapon_id_index = rearrange(
            repeat(vis_tensor[:, weapon_id_cols], 'b p -> b (p t)', t=num_radial_ticks).int(),
            'b pt -> (b pt)', pt=player_time_steps)
        flattened_scoped_index = rearrange(
            repeat(vis_tensor[:, scoped_cols], 'b p -> b (p t)', t=num_radial_ticks).int(),
            'b pt -> (b pt)', pt=player_time_steps)
        # mul weapon index by 2 as inner dimension is scoped, and 2 options for scoping (scoped or unscoped)
        flattened_max_speed_per_weapon_scoped = torch.index_select(
            weapon_scoped_to_max_speed, 0, 2 * flattened_weapon_id_index + flattened_scoped_index)
        flattened_max_speed_per_stature_weapon_scoped = \
            flattened_max_speed_per_weapon_scoped * flattened_max_speed_per_stature
    else:
        flattened_max_speed_per_stature_weapon_scoped = \
            max_speed_per_second * flattened_max_speed_per_stature
    per_batch_max_speed_per_stature = rearrange(flattened_max_speed_per_stature_weapon_scoped, '(b pt) -> b pt',
                                                pt=len(specific_player_place_area_columns) * num_radial_ticks)
    # scale by time per tick
    per_batch_max_speed_per_stature *= max_run_speed_per_sim_tick / max_speed_per_second
    # return cos/sin of dir degree
    not_moving_zeros = torch.zeros_like(per_batch_max_speed_per_stature)
    delta_pos = torch.stack([
        torch.where(not_moving, not_moving_zeros, torch.cos(torch.deg2rad(dir_degrees)) * per_batch_max_speed_per_stature),
        torch.where(not_moving, not_moving_zeros, torch.sin(torch.deg2rad(dir_degrees)) * per_batch_max_speed_per_stature),
        not_moving_zeros
    ], dim=-1)
    return DeltaPosWithZIndex(delta_pos, z_jump_index)


def compute_new_pos(input_pos_tensor: torch.Tensor, vis_tensor: torch.Tensor, pred_labels: torch.Tensor,
                    nav_data: NavData, pred_is_grid: bool,
                    stature_to_speed: torch.Tensor, weapon_scoped_to_max_speed: torch.Tensor):
    if pred_is_grid:
        delta_xyz_indices = get_delta_indices_from_grid(pred_labels)
        z_jump_index = delta_xyz_indices.z_jump_index
        z_index = torch.zeros_like(delta_xyz_indices.x_index)

        # convert to xy pos changes
        pos_index = torch.stack([delta_xyz_indices.x_index, delta_xyz_indices.y_index, z_index], dim=-1)
        # scaling is for simulator, not model
        unscaled_pos_change = (pos_index * delta_pos_grid_cell_dim)
        pos_change_norm = torch.linalg.vector_norm(unscaled_pos_change, dim=-1, keepdim=True)
        all_scaled_pos_change = (max_run_speed_per_sim_tick / pos_change_norm) * unscaled_pos_change
        # if norm less than max run speed, don't scale
        scaled_pos_change = torch.where(pos_change_norm > max_run_speed_per_sim_tick, all_scaled_pos_change,
                                        unscaled_pos_change)
    else:
        delta_pos_with_z = get_delta_pos_from_radial(pred_labels, vis_tensor, stature_to_speed,
                                                     weapon_scoped_to_max_speed)
        ## since this is for sim, and simulator steps one tick at a time (rather than 500ms like prediction), rescale it
        #unscaled_pos_change = delta_pos_with_z.delta_pos
        #pos_change_norm = torch.linalg.vector_norm(unscaled_pos_change, dim=-1, keepdim=True)
        #all_scaled_pos_change = (max_run_speed_per_sim_tick / pos_change_norm) * unscaled_pos_change
        ## if norm less than max run speed, don't scale
        #scaled_pos_change = torch.where(pos_change_norm > max_run_speed_per_sim_tick, all_scaled_pos_change,
        #                                unscaled_pos_change)
        scaled_pos_change = delta_pos_with_z.delta_pos
        # z is treated differently as need to look at navmesh
        z_jump_index = delta_pos_with_z.z_jump_index

    # apply to input pos
    # only want next pos change
    next_scaled_pos_change = rearrange(scaled_pos_change, 'b (p t) d -> b p t d',
                                       p=len(specific_player_place_area_columns), t=num_radial_ticks)[:, :, 0, :]
    next_z_jump_index = rearrange(z_jump_index, 'b (p t) -> b p t',
                                  p=len(specific_player_place_area_columns), t=num_radial_ticks)[:, :, 0]
    output_pos_tensor = input_pos_tensor[:, :, 0, :] + next_scaled_pos_change
    output_pos_tensor[:, :, 2] += torch.where(next_z_jump_index == 1., max_jump_height, 0.)
    output_pos_tensor[:, :, 0] = output_pos_tensor[:, :, 0].clamp(min=nav_data.nav_region.min.x,
                                                                  max=nav_data.nav_region.max.x)
    output_pos_tensor[:, :, 1] = output_pos_tensor[:, :, 1].clamp(min=nav_data.nav_region.min.y,
                                                                  max=nav_data.nav_region.max.y)
    output_pos_tensor[:, :, 2] = output_pos_tensor[:, :, 2].clamp(min=nav_data.nav_region.min.z,
                                                                  max=nav_data.nav_region.max.z)

    # compute z pos
    # number of steps in each dimension
    nav_grid_steps = torch.floor((output_pos_tensor - nav_data.nav_grid_base) / nav_step_size).long()
    # convert steps per dimension to linear index, and flatten since index_select accepts 1d array
    nav_grid_index = rearrange(torch.sum(nav_data.num_steps * nav_grid_steps, dim=-1), 'b p -> (b p)')
    nav_below_or_in_per_player = rearrange(torch.index_select(nav_data.nav_below_or_in, 0, nav_grid_index),
                                           '(b p) -> b p', p=len(specific_player_place_area_columns))
    output_pos_tensor[:, :, 2] = nav_below_or_in_per_player
    output_and_history_pos_tensor = torch.roll(input_pos_tensor, 1, 2)
    output_and_history_pos_tensor[:, :, 0, :] = output_pos_tensor
    return output_and_history_pos_tensor


def one_hot_max_to_index(pred: torch.Tensor) -> torch.Tensor:
    return torch.argmax(pred, 3, keepdim=True)


def one_hot_prob_to_index(pred: torch.Tensor) -> torch.Tensor:
    if pred.device.type == CUDA_DEVICE_STR:
        probs = rearrange(pred, 'b p t d -> (b p t) d', p=len(specific_player_place_area_columns), t=num_radial_ticks)
        return rearrange(torch.multinomial(probs, 1, replacement=True), '(b p t) d -> b p t d',
                         p=len(specific_player_place_area_columns), t=num_radial_ticks)
    else:
        return one_hot_max_to_index(pred)
