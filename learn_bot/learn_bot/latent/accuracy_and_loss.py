from typing import Callable

import torch
from torch.utils.tensorboard import SummaryWriter

from learn_bot.latent.dataset import *
from learn_bot.libs.io_transforms import IOColumnTransformers, ColumnTransformerType, CPU_DEVICE_STR, \
    get_untransformed_outputs, get_transformed_outputs
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
        self.cat_loss = torch.zeros([1])

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
def compute_loss(x, pred, y_transformed, y_untransformed, column_transformers: IOColumnTransformers,
                 latent_to_prob: Callable):
    x = x.to(CPU_DEVICE_STR)
    pred_transformed = get_transformed_outputs(pred)
    pred_transformed = pred_transformed.to(CPU_DEVICE_STR)
    pred_untransformed = get_untransformed_outputs(pred)
    pred_untransformed = pred_untransformed.to(CPU_DEVICE_STR)
    y_transformed = y_transformed.to(CPU_DEVICE_STR)
    y_untransformed = y_untransformed.to(CPU_DEVICE_STR)

    losses = LatentLosses()

    if column_transformers.output_types.categorical_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, frozenset({ColumnTransformerType.CATEGORICAL}))
        for i, col_range in enumerate(col_ranges):
            losses.cat_loss += base_classification_loss_fn(pred_transformed[:, col_range], y_transformed[:, col_range])

    if column_transformers.output_types.categorical_distribution_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, frozenset({ColumnTransformerType.CATEGORICAL_DISTRIBUTION}))
        # remove negatives where player isn't alive
        for i, col_range in enumerate(col_ranges):
            valid_y_transformed = y_transformed[y_transformed[:, col_range[0]] >= 0.][:, col_range]
            valid_pred_transformed = pred_transformed[y_transformed[:, col_range[0]] >= 0.][:, col_range]
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
    pred_untransformed = pred_untransformed.to(CPU_DEVICE_STR)
    Y = Y.to(CPU_DEVICE_STR)

    for name, col_range in zip(column_transformers.output_types.categorical_distribution_first_sub_cols,
                               column_transformers.get_name_ranges(False, False,
                                                                   frozenset({ColumnTransformerType.CATEGORICAL_DISTRIBUTION}))):
        valid_Y = Y[Y[:, col_range[0]] >= 0.][:, col_range]
        valid_pred_untransformed = pred_untransformed[Y[:, col_range[0]] >= 0.][:, col_range]
        if name not in valids_per_accuracy_column:
            valids_per_accuracy_column[name] = 0
        if valid_Y.shape[0] > 0:
            accuracy[name] += (torch.argmax(valid_pred_untransformed, -1, keepdim=True) ==
                               torch.argmax(valid_Y, -1, keepdim=True)).type(torch.float).sum().item()
            valids_per_accuracy_column[name] += len(valid_Y)


def finish_accuracy(accuracy, valids_per_accuracy_column, column_transformers: IOColumnTransformers):
    accuracy_string = ""
    for name, unadjusted_r in zip(column_transformers.output_types.column_names(True),
                                  column_transformers.get_name_ranges(False, False)):
        # make float accuracy into rmse
        if name in column_transformers.output_types.float_standard_cols or \
                name in column_transformers.output_types.delta_float_column_names() or \
                name in column_transformers.output_types.float_180_angle_cols or \
                name in column_transformers.output_types.delta_180_angle_column_names() or \
                name in column_transformers.output_types.float_90_angle_cols or \
                name in column_transformers.output_types.delta_90_angle_column_names():
            accuracy[name] = sqrt(accuracy[name])
            accuracy_string += f'''{name}: {accuracy[name]} rmse'''
        # record top-1 accuracy for others
        elif name in column_transformers.output_types.categorical_cols:
            accuracy_string += f'''{name}: {accuracy[name]} % cat top 1 acc'''
        elif name in column_transformers.output_types.column_names_all_categorical_columns():
            if valids_per_accuracy_column[name] > 0:
                accuracy_string += f'''{name}: {accuracy[name]} % cat top 1 acc'''
            else:
                accuracy_string += f'''{name}: no valids % cat top 1 acc'''
        else:
            raise "Invalid Column Type For finish_accuracy"
        accuracy_string += "; "
    return accuracy_string
