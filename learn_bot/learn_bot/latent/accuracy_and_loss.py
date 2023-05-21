from typing import Callable

import torch
from torch.utils.tensorboard import SummaryWriter

from learn_bot.latent.dataset import *
from learn_bot.libs.io_transforms import IOColumnTransformers, ColumnTransformerType, CPU_DEVICE_STR, \
    get_untransformed_outputs, get_transformed_outputs, CUDA_DEVICE_STR
from math import sqrt
from torch import nn

base_float_loss_fn = nn.HuberLoss(reduction='none')
def float_loss_fn(input, target, weight):
    return torch.sum(weight * base_float_loss_fn(input, target)) / torch.sum(weight)
base_binary_loss_fn = nn.BCEWithLogitsLoss()
def binary_loss_fn(input, target, weight):
    return torch.sum(weight * base_binary_loss_fn(input, target)) / torch.sum(weight)
# https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss
# no need to do softmax for classification output
base_classification_loss_fn = nn.CrossEntropyLoss()
def classification_loss_fn(input, target, weight, weight_sum):
    return (weight * base_classification_loss_fn(input, target)) / weight_sum


class LatentLosses:
    cat_loss: torch.Tensor

    def __init__(self):
        self.cat_loss = torch.zeros([1]).to(CUDA_DEVICE_STR)

    def get_total_loss(self):
        return self.cat_loss

    def __iadd__(self, other):
        self.cat_loss += other.cat_loss
        return self

    def __itruediv__(self, other):
        self.cat_loss /= other
        return self

    def add_scalars(self, writer: SummaryWriter, prefix: str, total_epoch_num: int):
        writer.add_scalar(prefix + '/loss/cat', self.cat_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/total', self.get_total_loss(), total_epoch_num)


# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, Y, column_transformers: IOColumnTransformers):
    pred_transformed = get_transformed_outputs(pred)

    losses = LatentLosses()

    if column_transformers.output_types.categorical_distribution_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, frozenset({ColumnTransformerType.CATEGORICAL_DISTRIBUTION}))
        # remove negatives where player isn't alive
        for i, col_range in enumerate(col_ranges):
            valid_rows = Y[:, col_range].sum(axis=1) > 0.1
            valid_y_transformed = Y[valid_rows][:, col_range]
            valid_pred_transformed = pred_transformed[valid_rows][:, col_range]
            # don't add nan loss if no valid rows
            if valid_y_transformed.shape[0] > 0:
                loss = base_classification_loss_fn(valid_pred_transformed, valid_y_transformed)
                if torch.isnan(loss).any():
                    print('bad loss')
                losses.cat_loss += loss
        #losses.cat_loss += latent_to_prob(pred_transformed, y_transformed, col_ranges)
    return losses


def compute_accuracy(pred, Y, accuracy, valids_per_accuracy_column, column_transformers: IOColumnTransformers):
    pred_untransformed = get_untransformed_outputs(pred)

    name = column_transformers.output_types.categorical_distribution_first_sub_cols[0]
    col_ranges = column_transformers.get_name_ranges(False, False, frozenset({ColumnTransformerType.CATEGORICAL_DISTRIBUTION}))

    Y_per_player = torch.unflatten(Y, 1, [-1, len(col_ranges[0])])
    pred_untransformed_per_player = torch.unflatten(pred_untransformed, 1, [-1, len(col_ranges[0])])
    accuracy_per_player = (torch.argmax(Y_per_player, -1) ==
                           torch.argmax(pred_untransformed_per_player, -1)).type(torch.float)
    Y_valid_per_player_row = Y_per_player.sum(axis=2)
    masked_accuracy_per_player = accuracy_per_player * Y_valid_per_player_row

    if name not in accuracy:
        accuracy[name] = 0
        valids_per_accuracy_column[name] = 0
    accuracy[name] += masked_accuracy_per_player.sum().item()
    valids_per_accuracy_column[name] += Y_valid_per_player_row.sum().item()


def finish_accuracy(accuracy, valids_per_accuracy_column, column_transformers: IOColumnTransformers):
    accuracy_string = ""
    for name, unadjusted_r in zip(column_transformers.output_types.column_names(True),
                                  column_transformers.get_name_ranges(False, False)):
        if name not in accuracy:
            continue
        elif name in column_transformers.output_types.column_names_all_categorical_columns():
            if valids_per_accuracy_column[name] > 0:
                accuracy_string += f'''{name}: {accuracy[name]} % cat top 1 acc'''
            else:
                accuracy_string += f'''{name}: no valids % cat top 1 acc'''
        accuracy_string += "; "
    return accuracy_string
