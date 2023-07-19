from typing import Callable

import torch
from einops import rearrange, repeat, pack
from torch.utils.tensorboard import SummaryWriter

from learn_bot.latent.dataset import *
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import get_delta_indices_from_grid, \
    get_delta_pos_from_radial
from learn_bot.libs.io_transforms import IOColumnTransformers, ColumnTransformerType, CPU_DEVICE_STR, \
    get_untransformed_outputs, get_transformed_outputs, CUDA_DEVICE_STR
from math import sqrt
from torch import nn

# https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss
# no need to do softmax for classification output
cross_entropy_loss_fn = nn.CrossEntropyLoss(reduction='none')

class LatentLosses:
    cat_loss: torch.Tensor
    cat_accumulator: float
    duplicate_last_cat_loss: torch.Tensor
    duplicate_last_cat_accumulator: float
    total_loss: torch.Tensor
    total_accumulator: float

    def __init__(self):
        self.cat_loss = torch.zeros([1]).to(CUDA_DEVICE_STR)
        self.cat_accumulator = 0.
        self.duplicate_last_cat_loss = torch.zeros([1]).to(CUDA_DEVICE_STR)
        self.duplicate_last_cat_accumulator = 0.
        self.total_loss = torch.zeros([1]).to(CUDA_DEVICE_STR)
        self.total_accumulator = 0.

    def __iadd__(self, other):
        self.cat_accumulator += other.cat_loss.item()
        self.duplicate_last_cat_accumulator += other.duplicate_last_cat_loss.item()
        self.total_accumulator += other.total_loss.item()
        return self

    def __itruediv__(self, other):
        self.cat_accumulator /= other
        self.duplicate_last_cat_accumulator /= other
        self.total_accumulator /= other
        return self

    def add_scalars(self, writer: SummaryWriter, prefix: str, total_epoch_num: int):
        writer.add_scalar(prefix + '/loss/cat', self.cat_accumulator, total_epoch_num)
        writer.add_scalar(prefix + '/loss/repeated cat', self.duplicate_last_cat_accumulator, total_epoch_num)
        writer.add_scalar(prefix + '/loss/total', self.total_accumulator, total_epoch_num)


# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, Y, duplicated_last, num_players) -> LatentLosses:
    pred_transformed = get_transformed_outputs(pred)

    losses = LatentLosses()

    # merging time steps so can do filtering before cross entropy loss
    Y_per_player = rearrange(Y, "b (p d) -> (b p) d", p=num_players)
    pred_transformed_per_player = rearrange(pred_transformed, "b (p d) -> (b p) d", p=num_players)
    duplicated_last_per_player = repeat(duplicated_last, "b -> (b repeat)", repeat=num_players)
    valid_rows = Y_per_player.sum(axis=1) > 0.1
    valid_Y_transformed = Y_per_player[valid_rows]
    valid_pred_transformed = pred_transformed_per_player[valid_rows]
    valid_duplicated_last_per_player = duplicated_last_per_player[valid_rows]
    #cat_loss = cross_entropy_loss_fn(valid_pred_transformed, valid_Y_transformed)
    #if torch.isnan(cat_loss).any():
    #    print('bad loss')
    #losses.cat_loss += cat_loss
    losses_to_cat = []
    if valid_Y_transformed[~valid_duplicated_last_per_player].shape[0] > 0:
        cat_loss = cross_entropy_loss_fn(valid_pred_transformed[~valid_duplicated_last_per_player],
                                         valid_Y_transformed[~valid_duplicated_last_per_player])
        if torch.isnan(cat_loss).any():
            print('bad loss')
        losses.cat_loss = torch.mean(cat_loss)
        losses_to_cat.append(cat_loss)
    if valid_Y_transformed[valid_duplicated_last_per_player].shape[0] > 0.:
        duplicated_last_cat_loss = cross_entropy_loss_fn(valid_pred_transformed[valid_duplicated_last_per_player],
                                                         valid_Y_transformed[valid_duplicated_last_per_player])
        if torch.isnan(duplicated_last_cat_loss).any():
            print('bad loss')
        losses.duplicate_last_cat_loss = torch.mean(duplicated_last_cat_loss)
        losses_to_cat.append(duplicated_last_cat_loss)
    #total_loss = cross_entropy_loss_fn(valid_pred_transformed, valid_Y_transformed)
    #if torch.isnan(total_loss).any():
    #    print('bad loss')
    #losses.total_loss = total_loss
    losses.total_loss = torch.mean(pack(losses_to_cat, '*')[0])

    return losses

duplicated_name_str = 'duplicated'

def compute_accuracy_and_delta_diff(pred, Y, duplicated_last, accuracy, delta_diff_xy, delta_diff_xyz,
                                    valids_per_accuracy_column, num_players, column_transformers: IOColumnTransformers,
                                    stature_to_speed):
    pred_untransformed = get_untransformed_outputs(pred)

    name = column_transformers.output_types.categorical_distribution_first_sub_cols[0]

    # keeping time steps flattened since just summing across all at end
    Y_per_player = rearrange(Y, "b (p d) -> b p d", p=num_players)
    Y_label_per_player = torch.argmax(Y_per_player, -1)
    pred_untransformed_per_player = rearrange(pred_untransformed, "b (p d) -> b p d", p=num_players)
    pred_untransformed_label_per_player = torch.argmax(pred_untransformed_per_player, -1)
    accuracy_per_player = (Y_label_per_player == pred_untransformed_label_per_player).type(torch.float)
    Y_valid_per_player_row = Y_per_player.sum(axis=2)
    masked_accuracy_per_player = accuracy_per_player * Y_valid_per_player_row

    # compute delta diffs
    #Y_delta_indices = get_delta_indices_from_grid(Y_label_per_player)
    #pred_untransformed_delta_indices = get_delta_indices_from_grid(pred_untransformed_label_per_player)
    Y_delta_pos = get_delta_pos_from_radial(Y_label_per_player, stature_to_speed)
    pred_untransformed_delta_pos = \
        get_delta_pos_from_radial(pred_untransformed_label_per_player, stature_to_speed)
    delta_diff_xy_per_player = \
        torch.sqrt(torch.sum(torch.pow(Y_delta_pos.delta_pos - pred_untransformed_delta_pos.delta_pos, 2), dim=-1))
    #torch.sqrt(
    #    torch.pow(Y_delta_indices.x_index - pred_untransformed_delta_indices.x_index, 2) +
    #    torch.pow(Y_delta_indices.y_index - pred_untransformed_delta_indices.y_index, 2)
    #)
    masked_delta_diff_xy = delta_diff_xy_per_player * Y_valid_per_player_row
    # 45 is reasonable jump height, so use that as z distance
    Y_delta_pos_with_z = Y_delta_pos.delta_pos.clone()
    Y_delta_pos_with_z[:, :, 2] = Y_delta_pos.z_jump_index * 45.
    pred_untransformed_delta_pos_with_z = pred_untransformed_delta_pos.delta_pos.clone()
    pred_untransformed_delta_pos_with_z[:, :, 2] = pred_untransformed_delta_pos.z_jump_index * 45.

    delta_diff_xyz_per_player = torch.sqrt(torch.sum(torch.pow(Y_delta_pos_with_z.delta_pos -
                                                               pred_untransformed_delta_pos_with_z.delta_pos, 2),
                                                     dim=-1))
    #    torch.sqrt(
    #    torch.pow(Y_delta_indices.x_index - pred_untransformed_delta_indices.x_index, 2) +
    #    torch.pow(Y_delta_indices.y_index - pred_untransformed_delta_indices.y_index, 2) +
    #    torch.pow(Y_delta_indices.z_jump_index - pred_untransformed_delta_indices.z_jump_index, 2)
    #)
    masked_delta_diff_xyz = delta_diff_xyz_per_player * Y_valid_per_player_row

    if name not in accuracy:
        accuracy[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xy[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xyz[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        valids_per_accuracy_column[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        accuracy[name + duplicated_name_str] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xy[name + duplicated_name_str] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xyz[name + duplicated_name_str] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        valids_per_accuracy_column[name + duplicated_name_str] = torch.zeros([1]).to(CUDA_DEVICE_STR)
    accuracy[name] += masked_accuracy_per_player[~duplicated_last].sum()
    delta_diff_xy[name] += masked_delta_diff_xy[~duplicated_last].sum()
    delta_diff_xyz[name] += masked_delta_diff_xyz[~duplicated_last].sum()
    valids_per_accuracy_column[name] += Y_valid_per_player_row[~duplicated_last].sum()
    accuracy[name + duplicated_name_str] += masked_accuracy_per_player[duplicated_last].sum()
    delta_diff_xy[name + duplicated_name_str] += masked_delta_diff_xy[duplicated_last].sum()
    delta_diff_xyz[name + duplicated_name_str] += masked_delta_diff_xyz[duplicated_last].sum()
    valids_per_accuracy_column[name + duplicated_name_str] += Y_valid_per_player_row[duplicated_last].sum()


def finish_accuracy_and_delta_diff(accuracy, delta_diff_xy, delta_diff_xyz, valids_per_accuracy_column,
                                   column_transformers: IOColumnTransformers):
    accuracy_string = ""
    for name, unadjusted_r in zip(column_transformers.output_types.column_names(True),
                                  column_transformers.get_name_ranges(False, False)):
        if name not in accuracy:
            continue
        elif name in column_transformers.output_types.column_names_all_categorical_columns():
            if valids_per_accuracy_column[name] > 0:
                accuracy_string += f"{name}: {accuracy[name]} % cat top 1 acc {delta_diff_xy[name]} delta diff xy " \
                                   f"{delta_diff_xyz[name]} delta diff xyz"
            else:
                accuracy_string += f'''{name}: no valids % cat top 1 acc'''
        accuracy_string += "; "
    return accuracy_string
