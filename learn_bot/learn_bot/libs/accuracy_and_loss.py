from learn_bot.engagement_aim.dataset import *
from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, ColumnTransformerType, CPU_DEVICE_STR, \
    CUDA_DEVICE_STR, get_untransformed_outputs, get_transformed_outputs
from math import sqrt
from torch import nn

float_loss_fn = nn.MSELoss(reduction='sum')
binary_loss_fn = nn.BCEWithLogitsLoss()
# https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss
# no need to do softmax for classification output
classification_loss_fn = nn.CrossEntropyLoss()


# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(pred, y, column_transformers: IOColumnTransformers):
    pred_transformed = get_transformed_outputs(pred)
    total_loss = 0
    if column_transformers.output_types.float_standard_cols or column_transformers.output_types.float_delta_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, frozenset({ColumnTransformerType.FLOAT_STANDARD,
                                                                                 ColumnTransformerType.FLOAT_DELTA}))
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)
        total_loss += float_loss_fn(pred_transformed[:, col_range], y[:, col_range])
    if column_transformers.output_types.categorical_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, frozenset({ColumnTransformerType.CATEGORICAL}))
        for col_range in col_ranges:
            total_loss += classification_loss_fn(pred_transformed[:, col_range], y[:, col_range])
    return total_loss


def compute_accuracy(pred, Y, accuracy, column_transformers: IOColumnTransformers):
    pred_untransformed = get_untransformed_outputs(pred)

    if column_transformers.output_types.float_standard_cols or column_transformers.output_types.float_delta_cols:
        col_ranges = column_transformers.get_name_ranges(False, False, frozenset({ColumnTransformerType.FLOAT_STANDARD,
                                                                                  ColumnTransformerType.FLOAT_DELTA}))
        col_range = range(col_ranges[0].start, col_ranges[-1].stop)
        squared_errors = torch.square(pred_untransformed[:, col_range] - Y[:, col_range]).sum(dim=0).to(CPU_DEVICE_STR)
        for i, name in enumerate(column_transformers.output_types.float_standard_cols +
                                 column_transformers.output_types.delta_float_column_names()):
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
                name in column_transformers.output_types.delta_float_column_names():
            accuracy[name] = sqrt(accuracy[name])
            accuracy_string += f'''{name}: {accuracy[name]} rmse'''
        # record top-1 accuracy for others
        elif name in column_transformers.output_types.categorical_cols:
            accuracy_string += f'''{name}: {accuracy[name]} % cat top 1 acc'''
        else:
            raise "Invalid Column Type For finish_accuracy"
        accuracy_string += "; "
    return accuracy_string
