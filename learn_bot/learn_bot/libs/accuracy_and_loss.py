import torch
from torch.utils.tensorboard import SummaryWriter

from learn_bot.engagement_aim.dataset import *
from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, ColumnTransformerType, CPU_DEVICE_STR, \
    CUDA_DEVICE_STR, get_untransformed_outputs, get_transformed_outputs
from math import sqrt
from torch import nn
from dataclasses import dataclass

#base_float_loss_fn = nn.MSELoss(reduction='sum')
base_float_loss_fn = nn.HuberLoss(reduction='none')
def float_loss_fn(input, target, weight):
    return torch.sum(weight * base_float_loss_fn(input, target)) / torch.sum(weight)
binary_loss_fn = nn.BCEWithLogitsLoss()
# https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss
# no need to do softmax for classification output
classification_loss_fn = nn.CrossEntropyLoss()


class AimLosses:
    pos_float_loss: torch.Tensor
    pos_attacking_float_loss: torch.Tensor
    target_float_loss: torch.Tensor
    cat_loss: torch.Tensor

    def __init__(self):
        self.pos_float_loss = torch.zeros([1])
        self.pos_attacking_float_loss = torch.zeros([1])
        self.target_float_loss = torch.zeros([1])
        self.speed_float_loss = torch.zeros([1])
        self.cat_loss = torch.zeros([1])

    def get_total_loss(self):
        return self.pos_float_loss + self.pos_attacking_float_loss + self.target_float_loss + self.cat_loss# + \
               #self.speed_float_loss + self.cat_loss

    def __iadd__(self, other):
        self.pos_float_loss += other.pos_float_loss
        self.pos_attacking_float_loss += other.pos_attacking_float_loss
        self.target_float_loss += other.target_float_loss
        self.speed_float_loss += other.speed_float_loss
        self.cat_loss += other.cat_loss
        return self

    def __itruediv__(self, other):
        self.pos_float_loss /= other
        self.pos_attacking_float_loss /= other
        self.target_float_loss /= other
        self.speed_float_loss /= other
        self.cat_loss /= other
        return self

    def add_scalars(self, writer: SummaryWriter, prefix: str, total_epoch_num: int):
        writer.add_scalar(prefix + '/loss/pos_float', self.pos_float_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/pos_attacking_float', self.pos_attacking_float_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/target_float', self.target_float_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/speed_float', self.speed_float_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/cat', self.cat_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/total', self.get_total_loss(), total_epoch_num)


def norm_2d(xy: torch.Tensor):
    return torch.sqrt(torch.pow(xy[:, :num_x_targets], 2) + torch.pow(xy[:, num_x_targets:], 2))


# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, y, transformed_targets, attacking, transformed_last_input_angles,
                 time_weights, column_transformers: IOColumnTransformers):
    pred_transformed = get_transformed_outputs(pred)
    pred_transformed = pred_transformed.to(CPU_DEVICE_STR)
    y = y.to(CPU_DEVICE_STR)
    transformed_targets = transformed_targets.to(CPU_DEVICE_STR)
    attacking = attacking.to(CPU_DEVICE_STR)
    # duplicate columns for yaw and pitch
    attacking_duplicated = torch.cat([attacking, attacking], dim=1)
    attacking_duplicated = torch.flatten(torch.unsqueeze(attacking_duplicated, -1).expand(-1, -1, 2), 1)
    time_weights_duplicated = torch.cat([time_weights, time_weights], dim=1)
    time_weights_duplicated = torch.flatten(torch.unsqueeze(time_weights_duplicated, -1).expand(-1, -1, 2), 1)
    #transformed_last_input_angles = transformed_last_input_angles.to(CPU_DEVICE_STR)
    #last_input_angles_x_duplicated = transformed_last_input_angles[:, [0]].expand(-1, time_weights.shape[1])
    #last_input_angles_y_duplicated = transformed_last_input_angles[:, [1]].expand(-1, time_weights.shape[1])
    #last_input_angles_duplicated = torch.cat([last_input_angles_x_duplicated, last_input_angles_y_duplicated], dim=1)

    losses = AimLosses()

    if column_transformers.output_types.float_standard_cols or column_transformers.output_types.float_delta_cols or \
        column_transformers.output_types.float_180_angle_cols or column_transformers.output_types.float_180_angle_delta_cols or \
        column_transformers.output_types.float_90_angle_cols or column_transformers.output_types.float_90_angle_delta_cols:
        col_ranges = column_transformers.get_name_ranges(False, True,
                                                         frozenset({ColumnTransformerType.FLOAT_STANDARD, ColumnTransformerType.FLOAT_DELTA,
                                                                    ColumnTransformerType.FLOAT_180_ANGLE, ColumnTransformerType.FLOAT_180_ANGLE_DELTA,
                                                                    ColumnTransformerType.FLOAT_90_ANGLE, ColumnTransformerType.FLOAT_90_ANGLE_DELTA}))
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)
        losses.pos_float_loss += float_loss_fn(pred_transformed[:, col_range], y[:, col_range], time_weights_duplicated)
        #losses.pos_attacking_float_loss += \
        #    float_loss_fn(pred_transformed[:, col_range] * attacking_duplicated, y[:, col_range] * attacking_duplicated,
        #                  time_weights_duplicated)

        #pred_target_distances = norm_2d((pred_transformed[:, col_range] - transformed_targets))
        #y_target_distances = norm_2d(y[:, col_range] - transformed_targets)
        #losses.target_float_loss += float_loss_fn(pred_target_distances, y_target_distances, time_weights)

        #pred_speed = norm_2d(pred_transformed[:, col_range] - last_input_angles_duplicated)
        #y_speed = norm_2d(y[:, col_range] - last_input_angles_duplicated)
        #losses.speed_float_loss += float_loss_fn(pred_speed, y_speed, time_weights)
    if column_transformers.output_types.categorical_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, frozenset({ColumnTransformerType.CATEGORICAL}))
        for col_range in col_ranges:
            losses.cat_loss += classification_loss_fn(pred_transformed[:, col_range], y[:, col_range])
    return losses


def compute_accuracy(pred, Y, accuracy, column_transformers: IOColumnTransformers):
    pred_untransformed = get_untransformed_outputs(pred)
    pred_untransformed = pred_untransformed.to(CPU_DEVICE_STR)
    Y = Y.to(CPU_DEVICE_STR)

    if column_transformers.output_types.float_standard_cols or column_transformers.output_types.float_delta_cols or \
            column_transformers.output_types.float_180_angle_cols or column_transformers.output_types.float_180_angle_delta_cols or \
            column_transformers.output_types.float_90_angle_cols or column_transformers.output_types.float_90_angle_delta_cols:
        col_ranges = column_transformers.get_name_ranges(False, False,
                                                         frozenset({ColumnTransformerType.FLOAT_STANDARD, ColumnTransformerType.FLOAT_DELTA,
                                                                    ColumnTransformerType.FLOAT_180_ANGLE, ColumnTransformerType.FLOAT_180_ANGLE_DELTA,
                                                                    ColumnTransformerType.FLOAT_90_ANGLE, ColumnTransformerType.FLOAT_90_ANGLE_DELTA}))
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)
        squared_errors = torch.square(pred_untransformed[:, col_range] - Y[:, col_range]).sum(dim=0).to(CPU_DEVICE_STR)
        for i, name in enumerate(column_transformers.output_types.float_standard_cols +
                                 column_transformers.output_types.delta_float_column_names() +
                                 column_transformers.output_types.float_180_angle_cols +
                                 column_transformers.output_types.delta_180_angle_column_names() +
                                 column_transformers.output_types.float_90_angle_cols +
                                 column_transformers.output_types.delta_90_angle_column_names()
                                 ):
            accuracy[name] += squared_errors[i].item()

    for name, col_range in zip(column_transformers.output_types.categorical_cols,
                               column_transformers.get_name_ranges(False, False,
                                                                   frozenset({ColumnTransformerType.CATEGORICAL}))):
        # compute accuracy using unnormalized outputs on end
        accuracy[name] += (pred_untransformed[:, col_range] == Y[:, col_range]).type(torch.float).sum().item()


def finish_accuracy(accuracy, column_transformers: IOColumnTransformers):
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
        else:
            raise "Invalid Column Type For finish_accuracy"
        accuracy_string += "; "
    return accuracy_string
