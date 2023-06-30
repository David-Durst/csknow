from typing import Callable

import torch
from torch.utils.tensorboard import SummaryWriter

from learn_bot.latent.dataset import *
from learn_bot.latent.place_area.pos_abs_delta_conversion import get_delta_indices
from learn_bot.libs.io_transforms import IOColumnTransformers, ColumnTransformerType, CPU_DEVICE_STR, \
    get_untransformed_outputs, get_transformed_outputs, CUDA_DEVICE_STR
from math import sqrt
from torch import nn

# https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss
# no need to do softmax for classification output
cross_entropy_loss_fn = nn.CrossEntropyLoss()


# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, Y, column_transformers: IOColumnTransformers) -> torch.Tensor:
    pred_transformed = get_transformed_outputs(pred)


    col_ranges = column_transformers.get_name_ranges(False, False, frozenset({ColumnTransformerType.CATEGORICAL_DISTRIBUTION}))

    # merging time steps so can do filtering before cross entropy loss
    Y_per_player = torch.flatten(torch.unflatten(Y, 1, [-1, len(col_ranges[0])]), 0, 1)
    pred_transformed_per_player = torch.flatten(torch.unflatten(pred_transformed, 1, [-1, len(col_ranges[0])]), 0, 1)
    valid_rows = Y_per_player.sum(axis=1) > 0.1
    valid_Y_transformed = Y_per_player[valid_rows]
    valid_pred_transformed = pred_transformed_per_player[valid_rows]
    if valid_Y_transformed.shape[0] > 0:
        loss = cross_entropy_loss_fn(valid_pred_transformed, valid_Y_transformed)
        if torch.isnan(loss).any():
            print('bad loss')
        return loss
    else:
        torch.zeros([1]).to(CUDA_DEVICE_STR)


def compute_accuracy_and_delta_diff(pred, Y, accuracy, delta_diff_xy, delta_diff_xyz, valids_per_accuracy_column,
                                    column_transformers: IOColumnTransformers):
    pred_untransformed = get_untransformed_outputs(pred)

    name = column_transformers.output_types.categorical_distribution_first_sub_cols[0]
    col_ranges = column_transformers.get_name_ranges(False, False, frozenset({ColumnTransformerType.CATEGORICAL_DISTRIBUTION}))

    # keeping time steps flattened since just summing across all at end
    Y_per_player = torch.unflatten(Y, 1, [-1, len(col_ranges[0])])
    Y_label_per_player = torch.argmax(Y_per_player, -1)
    pred_untransformed_per_player = torch.unflatten(pred_untransformed, 1, [-1, len(col_ranges[0])])
    pred_untransformed_label_per_player = torch.argmax(pred_untransformed_per_player, -1)
    accuracy_per_player = (Y_label_per_player == pred_untransformed_label_per_player).type(torch.float)
    Y_valid_per_player_row = Y_per_player.sum(axis=2)
    masked_accuracy_per_player = accuracy_per_player * Y_valid_per_player_row

    # compute delta diffs
    Y_delta_indices = get_delta_indices(Y_label_per_player)
    pred_untransformed_delta_indices = get_delta_indices(pred_untransformed_label_per_player)
    delta_diff_xy_per_player = torch.sqrt(
        torch.pow(Y_delta_indices.x_index - pred_untransformed_delta_indices.x_index, 2) +
        torch.pow(Y_delta_indices.y_index - pred_untransformed_delta_indices.y_index, 2)
    )
    masked_delta_diff_xy = delta_diff_xy_per_player * Y_valid_per_player_row
    delta_diff_xyz_per_player = torch.sqrt(
        torch.pow(Y_delta_indices.x_index - pred_untransformed_delta_indices.x_index, 2) +
        torch.pow(Y_delta_indices.y_index - pred_untransformed_delta_indices.y_index, 2) +
        torch.pow(Y_delta_indices.z_jump_index - pred_untransformed_delta_indices.z_jump_index, 2)
    )
    masked_delta_diff_xyz = delta_diff_xyz_per_player * Y_valid_per_player_row

    if name not in accuracy:
        accuracy[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xy[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xyz[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        valids_per_accuracy_column[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
    accuracy[name] += masked_accuracy_per_player.sum()
    delta_diff_xy[name] += masked_delta_diff_xy.sum()
    delta_diff_xyz[name] += masked_delta_diff_xyz.sum()
    valids_per_accuracy_column[name] += Y_valid_per_player_row.sum()


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
