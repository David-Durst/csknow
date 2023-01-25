import torch
from torch.utils.tensorboard import SummaryWriter

from learn_bot.engagement_aim.dataset import *
from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, ColumnTransformerType, CPU_DEVICE_STR, \
    CUDA_DEVICE_STR, get_untransformed_outputs, get_transformed_outputs, PT180AngleColumnTransformer, \
    PT90AngleColumnTransformer
from math import sqrt
from torch import nn
from dataclasses import dataclass

#base_float_loss_fn = nn.MSELoss(reduction='sum')
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


class AimLosses:
    pos_float_loss: torch.Tensor
    pos_cos_sin_loss: torch.Tensor
    pos_attacking_float_loss: torch.Tensor
    target_float_loss: torch.Tensor
    cat_loss: torch.Tensor

    def __init__(self):
        self.pos_float_loss = torch.zeros([1])
        self.pos_sin_cos_loss = torch.zeros([1])
        self.pos_attacking_float_loss = torch.zeros([1])
        self.target_float_loss = torch.zeros([1])
        self.speed_float_loss = torch.zeros([1])
        self.cat_loss = torch.zeros([1])

    def get_total_loss(self):
        return self.pos_float_loss + self.pos_sin_cos_loss + self.pos_attacking_float_loss + self.target_float_loss / 200 + self.cat_loss# + \
               #self.speed_float_loss + self.cat_loss

    def __iadd__(self, other):
        self.pos_float_loss += other.pos_float_loss
        self.pos_sin_cos_loss += other.pos_sin_cos_loss
        self.pos_attacking_float_loss += other.pos_attacking_float_loss
        self.target_float_loss += other.target_float_loss
        self.speed_float_loss += other.speed_float_loss
        self.cat_loss += other.cat_loss
        return self

    def __itruediv__(self, other):
        self.pos_float_loss /= other
        self.pos_sin_cos_loss /= other
        self.pos_attacking_float_loss /= other
        self.target_float_loss /= other
        self.speed_float_loss /= other
        self.cat_loss /= other
        return self

    def add_scalars(self, writer: SummaryWriter, prefix: str, total_epoch_num: int):
        writer.add_scalar(prefix + '/loss/pos_float', self.pos_float_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/pos_sin_cos', self.pos_sin_cos_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/pos_attacking_float', self.pos_attacking_float_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/target_float', self.target_float_loss / 200, total_epoch_num)
        writer.add_scalar(prefix + '/loss/speed_float', self.speed_float_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/cat', self.cat_loss, total_epoch_num)
        writer.add_scalar(prefix + '/loss/total', self.get_total_loss(), total_epoch_num)


def norm_2d(xy: torch.Tensor):
    return torch.sqrt(torch.pow(xy[:, :num_x_targets], 2) + torch.pow(xy[:, num_x_targets:], 2))


def wrap_angles(angle_deltas: torch.Tensor) -> torch.Tensor:
    sin_deltas = torch.sin(torch.deg2rad(angle_deltas))
    cos_deltas = torch.cos(torch.deg2rad(angle_deltas))
    return torch.rad2deg(torch.atan2(sin_deltas, cos_deltas))

# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(x, pred, y_transformed, y_untransformed, targets, attacking, transformed_last_input_angles,
                 time_weights, column_transformers: IOColumnTransformers, include_cat_cols):
    x = x.to(CPU_DEVICE_STR)
    pred_transformed = get_transformed_outputs(pred)
    pred_transformed = pred_transformed.to(CPU_DEVICE_STR)
    pred_untransformed = get_untransformed_outputs(pred)
    pred_untransformed = pred_untransformed.to(CPU_DEVICE_STR)
    y_transformed = y_transformed.to(CPU_DEVICE_STR)
    y_untransformed = y_untransformed.to(CPU_DEVICE_STR)
    targets = targets.to(CPU_DEVICE_STR)
    attacking = attacking.to(CPU_DEVICE_STR)
    # duplicate columns for yaw and pitch
    attacking_duplicated = torch.cat([attacking, attacking], dim=1)
    attacking_duplicated = torch.flatten(torch.unsqueeze(attacking_duplicated, -1).expand(-1, -1, 2), 1)
    time_weights_sin_cos = torch.flatten(torch.unsqueeze(time_weights, -1).expand(-1, -1, 2), 1)
    time_weights_duplicated = torch.cat([time_weights, time_weights], dim=1)
    time_weights_duplicated_sin_cos = torch.cat([time_weights_sin_cos, time_weights_sin_cos], dim=1)
    #transformed_last_input_angles = transformed_last_input_angles.to(CPU_DEVICE_STR)
    #last_input_angles_x_duplicated = transformed_last_input_angles[:, [0]].expand(-1, time_weights.shape[1])
    #last_input_angles_y_duplicated = transformed_last_input_angles[:, [1]].expand(-1, time_weights.shape[1])
    #last_input_angles_duplicated = torch.cat([last_input_angles_x_duplicated, last_input_angles_y_duplicated], dim=1)

    losses = AimLosses()

    if column_transformers.output_types.float_standard_cols or column_transformers.output_types.float_delta_cols:
        col_ranges = column_transformers.get_name_ranges(False, False,
                                                         frozenset({ColumnTransformerType.FLOAT_STANDARD,
                                                                    ColumnTransformerType.FLOAT_DELTA}))
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)
        losses.pos_float_loss += float_loss_fn(pred_untransformed[:, col_range], y_untransformed[:, col_range],
                                               time_weights.repeat(1, int(len(col_range) / time_weights.shape[1])))
        #col_range = range(col_ranges[0].start, col_ranges[-1].stop)
        #losses.pos_float_loss += float_loss_fn(pred_transformed[:, col_range], y_transformed[:, col_range],
        #                                       time_weights.repeat(1, int(len(col_range) / time_weights.shape[1])))
        # losses.pos_attacking_float_loss += \
        #    float_loss_fn(pred_transformed[:, col_range] * attacking_duplicated,
        #                  y_transformed[:, col_range] * attacking_duplicated,
        #                  time_weights_duplicated)

        #col_ranges_untransformed = column_transformers.get_name_ranges(False, False,
        #                                                               frozenset({ColumnTransformerType.FLOAT_STANDARD,
        #                                                                          ColumnTransformerType.FLOAT_DELTA}))
        #col_range_untransformed = range(col_ranges_untransformed[0].start, col_ranges_untransformed[-1].stop)
        #pred_target_distances = norm_2d(pred_untransformed[:, col_range_untransformed] - targets[num_x_targets:]) / 180.
        #y_target_distances = norm_2d(y_untransformed[:, col_range_untransformed] - targets[num_x_targets:]) / 180.
        #losses.target_float_loss += float_loss_fn(pred_target_distances, y_target_distances, time_weights)

        # pred_speed = norm_2d(pred_transformed[:, col_range] - last_input_angles_duplicated)
        # y_speed = norm_2d(y_transformed[:, col_range] - last_input_angles_duplicated)
        # losses.speed_float_loss += float_loss_fn(pred_speed, y_speed, time_weights)
    if column_transformers.output_types.float_180_angle_cols or column_transformers.output_types.float_180_angle_delta_cols or \
            column_transformers.output_types.float_90_angle_cols or column_transformers.output_types.float_90_angle_delta_cols:

        # compute pos loss
        col_ranges = column_transformers.get_name_ranges(False, True,
                                                         frozenset({ColumnTransformerType.FLOAT_180_ANGLE,
                                                                    ColumnTransformerType.FLOAT_180_ANGLE_DELTA,
                                                                    ColumnTransformerType.FLOAT_90_ANGLE,
                                                                    ColumnTransformerType.FLOAT_90_ANGLE_DELTA}))
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)

        #wrapped = wrap_angles(pred_untransformed[:, col_range] - y_untransformed[:, col_range])
        #losses.pos_sin_cos_loss += float_loss_fn(wrapped, torch.zeros_like(wrapped),
        #                                         time_weights_duplicated)
        losses.pos_sin_cos_loss += float_loss_fn(pred_transformed[:, col_range], y_transformed[:, col_range],
                                                 time_weights_duplicated_sin_cos)

        # compute distance to target loss
        if False:
            col_ranges_180 = column_transformers.get_name_ranges(False, False,
                                                                 frozenset({ColumnTransformerType.FLOAT_180_ANGLE,
                                                                            ColumnTransformerType.FLOAT_180_ANGLE_DELTA}))
            col_range_180 = range(col_ranges_180[0].start, col_ranges_180[-1].stop)

            wrapped_pred_target_differences_180 = wrap_angles(pred_untransformed[:, col_range_180] -
                                                              targets[:, col_range_180])

            wrapped_y_target_differences_180 = wrap_angles(y_untransformed[:, col_range_180] -
                                                           targets[:, col_range_180])

            col_ranges_90 = column_transformers.get_name_ranges(False, False,
                                                                 frozenset({ColumnTransformerType.FLOAT_90_ANGLE,
                                                                            ColumnTransformerType.FLOAT_90_ANGLE_DELTA}))
            col_range_90 = range(col_ranges_90[0].start, col_ranges_90[-1].stop)

            pred_target_differences_90 = (pred_untransformed[:, col_range_90] - targets[:, col_range_90])
            y_target_differences_90 = (y_untransformed[:, col_range_90] - targets[:, col_range_90])

            pred_target_distances = norm_2d(torch.cat([wrapped_pred_target_differences_180, pred_target_differences_90],
                                                      dim=1))
            y_target_distances = norm_2d(torch.cat([wrapped_y_target_differences_180, y_target_differences_90],
                                                   dim=1))
            losses.target_float_loss += float_loss_fn(pred_target_distances, y_target_distances, time_weights)
        #losses.pos_float_loss += float_loss_fn(fixed_angular_differences, torch.zeros_like(fixed_angular_differences),
        #                                       time_weights) / 180.
    #if column_transformers.output_types.float_90_angle_cols or column_transformers.output_types.float_90_angle_delta_cols:
    #    col_ranges = column_transformers.get_name_ranges(False, False,
    #                                                     frozenset({ColumnTransformerType.FLOAT_90_ANGLE,
    #                                                                ColumnTransformerType.FLOAT_90_ANGLE_DELTA}))
    #    col_range = range(col_ranges[0].start, col_ranges[-1].stop)
    #    angular_differences = pred_untransformed[:, col_range] - y_untransformed[:, col_range]
    #    fixed_angular_differences = angle_transformer_90.inverse(angle_transformer_90.convert(angular_differences))
    #    losses.pos_float_loss += float_loss_fn(fixed_angular_differences, torch.zeros_like(fixed_angular_differences),
    #                                           time_weights) / 90.
    if include_cat_cols and column_transformers.output_types.categorical_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, frozenset({ColumnTransformerType.CATEGORICAL}))
        for i, col_range in enumerate(col_ranges):
            losses.cat_loss += classification_loss_fn(pred_transformed[:, col_range], y_transformed[:, col_range],
                                                      time_weights[0, i % len(time_weights[0])], torch.sum(time_weights))
    return losses


def compute_accuracy(pred, Y, accuracy, column_transformers: IOColumnTransformers):
    pred_untransformed = get_untransformed_outputs(pred)
    pred_untransformed = pred_untransformed.to(CPU_DEVICE_STR)
    Y = Y.to(CPU_DEVICE_STR)

    if column_transformers.output_types.float_standard_cols or column_transformers.output_types.float_delta_cols or \
            column_transformers.output_types.float_90_angle_cols or column_transformers.output_types.float_90_angle_delta_cols:
        # don't wrap 90 as can't wrap vertically in csgo
        col_ranges = column_transformers.get_name_ranges(False, False,
                                                         frozenset({ColumnTransformerType.FLOAT_STANDARD,
                                                                    ColumnTransformerType.FLOAT_DELTA,
                                                                    ColumnTransformerType.FLOAT_90_ANGLE,
                                                                    ColumnTransformerType.FLOAT_90_ANGLE_DELTA}))
        squared_errors = torch.square(torch.flatten(pred_untransformed[:, col_ranges] - Y[:, col_ranges], start_dim=1)) \
            .sum(dim=0).to(CPU_DEVICE_STR)
        for i, name in enumerate(column_transformers.output_types.float_standard_cols +
                                 column_transformers.output_types.delta_float_column_names() +
                                 column_transformers.output_types.float_90_angle_cols +
                                 column_transformers.output_types.delta_90_angle_column_names()):
            accuracy[name] += squared_errors[i].item()
    if column_transformers.output_types.float_180_angle_cols or \
            column_transformers.output_types.float_180_angle_delta_cols:
        col_ranges = column_transformers.get_name_ranges(False, False,
                                                         frozenset({ColumnTransformerType.FLOAT_180_ANGLE,
                                                                    ColumnTransformerType.FLOAT_180_ANGLE_DELTA}))
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)

        wrapped_angle_differences = wrap_angles(pred_untransformed[:, col_range] - Y[:, col_range])
        squared_errors = torch.square(wrapped_angle_differences).sum(dim=0).to(CPU_DEVICE_STR)
        for i, name in enumerate(column_transformers.output_types.float_180_angle_cols +
                                 column_transformers.output_types.delta_180_angle_column_names()):
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
