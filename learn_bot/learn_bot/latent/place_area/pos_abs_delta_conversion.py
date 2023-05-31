from math import isqrt, ceil
from pathlib import Path

import torch
from einops import rearrange

from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, delta_pos_grid_radius, \
    delta_pos_grid_cell_dim, delta_pos_z_num_cells, delta_pos_grid_num_cells, delta_pos_grid_num_cells_per_xy_dim, \
    delta_pos_grid_num_xy_cells_per_z_change
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
max_run_speed_per_tick = 130. / (12.8 / 2)
max_jump_height = 65.
nav_step_size = 10.


def compute_new_pos(input_pos_tensor: torch.tensor, pred_labels: torch.Tensor, nav_data: NavData):
    # compute indices for changes
    z_jump_index = torch.floor(pred_labels / delta_pos_grid_num_xy_cells_per_z_change)
    xy_pred_label = torch.floor(torch.remainder(pred_labels, delta_pos_grid_num_xy_cells_per_z_change))
    x_index = torch.floor(torch.remainder(xy_pred_label, delta_pos_grid_num_cells_per_xy_dim)) - \
              int(delta_pos_grid_num_cells_per_xy_dim / 2)
    y_index = torch.floor(xy_pred_label / delta_pos_grid_num_cells_per_xy_dim) - \
              int(delta_pos_grid_num_cells_per_xy_dim / 2)
    z_index = torch.zeros_like(x_index)


    # convert to xy pos changes
    pos_index = torch.stack([x_index, y_index, z_index], dim=-1)
    unscaled_pos_change = (pos_index * delta_pos_grid_cell_dim)
    pos_change_norm = torch.linalg.vector_norm(unscaled_pos_change, dim=-1, keepdim=True)
    all_scaled_pos_change = (max_run_speed_per_tick / pos_change_norm) * unscaled_pos_change
    # if norm less than max run speed, don't scale
    scaled_pos_change = torch.where(pos_change_norm > max_run_speed_per_tick, all_scaled_pos_change,
                                    unscaled_pos_change)

    # apply to input pos
    output_pos_tensor = input_pos_tensor + scaled_pos_change
    output_pos_tensor[:, :, 2] += torch.where(z_jump_index == 2., max_jump_height, 0.)
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
    output_pos_tensor[:, :, 2] = torch.where(z_jump_index == 1,
                                             nav_above_below_per_player[:, :, 0], nav_above_below_per_player[:, :, 1])
    return rearrange(output_pos_tensor, 'b p d -> b (p d)')


def delta_one_hot_max_to_index(pred: torch.Tensor) -> torch.Tensor:
    return torch.argmax(rearrange(pred, 'b (p d) -> b p d', p=len(specific_player_place_area_columns)), 2)


def delta_one_hot_prob_to_index(pred: torch.Tensor) -> torch.Tensor:
    if pred.device.type == CUDA_DEVICE_STR:
        probs = rearrange(pred, 'b (p d) -> (b p) d', p=len(specific_player_place_area_columns))
        return rearrange(torch.multinomial(probs, 1, replacement=True), '(b p) d -> b (p d)',
                         p=len(specific_player_place_area_columns))
    else:
        return delta_one_hot_max_to_index(pred)