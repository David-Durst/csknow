from learn_bot.engagement_aim.dataset import *
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTransformerType, CPU_DEVICE_STR, CUDA_DEVICE_STR
from math import sqrt
from torch import nn

float_loss_fn = nn.MSELoss(reduction='sum')
binary_loss_fn = nn.BCEWithLogitsLoss()
classification_loss_fn = nn.CrossEntropyLoss()


# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, y, column_transformers: IOColumnTransformers):
    total_loss = 0
    if column_transformers.output_types.float_standard_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, {ColumnTransformerType.FLOAT_STANDARD})
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)
        total_loss += float_loss_fn(pred[:, col_range], y[:, col_range])
    if column_transformers.output_types.categorical_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, {ColumnTransformerType.CATEGORICAL})
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)
        total_loss += classification_loss_fn(pred[:, col_range], y[:, col_range])
    return total_loss


def compute_accuracy(pred, Y, accuracy, column_transformers: IOColumnTransformers):
    num_output_cols = len(column_transformers.output_types.column_names())

    if column_transformers.output_types.float_standard_cols:
        col_ranges = column_transformers.get_name_ranges(False, False, {ColumnTransformerType.FLOAT_STANDARD})
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)
        untransformed_col_ranges = range(col_range.start + num_output_cols, col_range.stop + num_output_cols)
        squared_errors = torch.square(pred[:, untransformed_col_ranges] - Y[:, col_range]).sum(dim=0).to(CPU_DEVICE_STR)
        for i, name in enumerate(column_transformers.output_types.float_standard_cols):
            accuracy[name] += squared_errors[i].item()

    for name, col_range in zip(column_transformers.output_types.categorical_cols,
                               column_transformers.get_name_ranges(False, False, {ColumnTransformerType.CATEGORICAL})):
        # compute accuracy using unnormalized outputs on end
        untransformed_col_range = range(col_range.start + num_output_cols, col_range.stop + num_output_cols)
        accuracy[name] += (pred[:, untransformed_col_range].argmax(1) == Y[:, col_range].argmax(1)) \
            .type(torch.float).sum().item()


def finish_accuracy(accuracy, column_transformers: IOColumnTransformers):
    accuracy_string = ""
    for name, unadjusted_r in zip(column_transformers.output_types.column_names(),
                                  column_transformers.get_name_ranges(False, False)):
        # make float accuracy into rmse
        if name in column_transformers.output_types.float_standard_cols or \
                name in column_transformers.output_types.float_min_max_cols or \
                name in column_transformers.output_types.float_non_linear_cols:
            accuracy[name] = sqrt(accuracy[name])
            accuracy_string += f'''{name}: {accuracy[name]} rmse'''
        # record top-1 accuracy for tohers
        elif name in column_transformers.output_types.boolean_cols:
            accuracy_string += f'''{name}: {accuracy[name]} bool eq'''
        elif name in column_transformers.output_types.categorical_cols:
            accuracy_string += f'''{name}: {accuracy[name]} % cat top 1 acc'''
        else:
            raise "Invalid Column Type For finish_accuracy"
        accuracy_string += "; "
    return accuracy_string
