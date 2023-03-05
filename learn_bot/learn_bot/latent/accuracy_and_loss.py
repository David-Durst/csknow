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
def compute_loss(x, pred, y_transformed, y_untransformed, column_transformers: IOColumnTransformers):
    x = x.to(CPU_DEVICE_STR)
    pred_transformed = get_transformed_outputs(pred)
    pred_transformed = pred_transformed.to(CPU_DEVICE_STR)
    y_transformed = y_transformed.to(CPU_DEVICE_STR)
    y_untransformed = y_untransformed.to(CPU_DEVICE_STR)

    losses = LatentLosses()

    if column_transformers.output_types.categorical_cols:
        col_ranges = column_transformers.get_name_ranges(False, True, frozenset({ColumnTransformerType.CATEGORICAL}))
        for i, col_range in enumerate(col_ranges):
            losses.cat_loss += base_classification_loss_fn(pred_transformed[:, col_range], y_transformed[:, col_range])
    return losses


def compute_accuracy(pred, Y, accuracy, column_transformers: IOColumnTransformers):
    pred_untransformed = get_untransformed_outputs(pred)
    pred_untransformed = pred_untransformed.to(CPU_DEVICE_STR)
    Y = Y.to(CPU_DEVICE_STR)

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
