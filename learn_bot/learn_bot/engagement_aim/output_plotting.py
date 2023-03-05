from pathlib import Path

import pandas as pd
from typing import List
from learn_bot.engagement_aim.accuracy_and_loss import CPU_DEVICE_STR

import torch

from learn_bot.engagement_aim.mlp_aim_model import MLPAimModel
from learn_bot.libs.io_transforms import IOColumnTransformers, get_transformed_outputs, \
    get_untransformed_outputs, ModelOutput
from learn_bot.libs.plot_features import plot_untransformed_and_transformed

plot_path = Path(__file__).parent / 'distributions'

#float_column_x_axes: List[str] = ['yaw degree', 'pitch degree', 'yaw degree', 'pitch degree',
#                                  'yaw degree', 'pitch degree',
#                                  'hammer units', 'hammer units', 'hammer units', 'hammer units']
#
#cat_column_x_axes: List[str] = ['weapon type']


class ModelOutputRecording:
    train_outputs_untransformed: torch.Tensor
    train_outputs_transformed: torch.Tensor
    test_outputs_untransformed: torch.Tensor
    test_outputs_transformed: torch.Tensor
    test_errors_untransformed: torch.Tensor
    test_errors_transformed: torch.Tensor
    model: MLPAimModel

    def __init__(self, model: MLPAimModel):
        self.train_outputs_untransformed = None
        self.train_outputs_transformed = None
        self.test_outputs_untransformed = None
        self.test_outputs_transformed = None
        self.test_errors_untransformed = None
        self.test_errors_transformed = None
        self.model = model

    def record_output(self, pred: ModelOutput, Y: torch.Tensor, transformed_Y: torch.Tensor, train):
        pred = (pred[0].detach(), pred[1].detach())
        Y = Y.detach()
        transformed_Y = transformed_Y.detach()

        if train:
            if self.train_outputs_untransformed is None:
                self.train_outputs_untransformed = get_untransformed_outputs(pred)
                self.train_outputs_transformed = get_transformed_outputs(pred)
            else:
                self.train_outputs_untransformed = torch.cat(
                    [self.train_outputs_untransformed, get_untransformed_outputs(pred)], 0)
                self.train_outputs_transformed = torch.cat(
                    [self.train_outputs_transformed, get_transformed_outputs(pred)], 0)
        else:
            if self.test_outputs_untransformed is None:
                self.test_outputs_untransformed = get_untransformed_outputs(pred)
                self.test_outputs_transformed = get_transformed_outputs(pred)
                self.test_errors_untransformed = get_untransformed_outputs(pred) - Y
                self.test_errors_transformed = get_transformed_outputs(pred) - transformed_Y
            else:
                self.test_outputs_untransformed = torch.cat(
                    [self.test_outputs_untransformed, get_untransformed_outputs(pred)], 0)
                self.test_outputs_transformed = torch.cat(
                    [self.test_outputs_transformed, get_transformed_outputs(pred)], 0)
                self.test_errors_untransformed = torch.cat(
                    [self.test_errors_untransformed, get_untransformed_outputs(pred) - Y], 0)
                self.test_errors_transformed = torch.cat(
                    [self.test_errors_transformed, get_transformed_outputs(pred) - transformed_Y], 0)

    def plot(self, cts: IOColumnTransformers, vis_float_columns: List[str], vis_cat_columns: List[str]):
        test_df = pd.DataFrame(self.test_outputs_untransformed.to(CPU_DEVICE_STR),
                               columns=cts.output_types.column_names())
        #transformed_test_df = pd.DataFrame(self.test_outputs_transformed.to(CPU_DEVICE_STR),
        #                                   columns=cts.output_types.column_names())
        plot_untransformed_and_transformed(plot_path, 'test predictions', test_df, vis_float_columns, vis_cat_columns)
                                           #transformed_test_df)

        test_errors_df = pd.DataFrame(self.test_errors_untransformed.to(CPU_DEVICE_STR),
                                      columns=cts.output_types.column_names())
        #transformed_test_errors_df = pd.DataFrame(self.test_errors_transformed.to(CPU_DEVICE_STR),
        #                                          columns=cts.output_types.column_names())
        plot_untransformed_and_transformed(plot_path, 'test errors', test_errors_df, vis_float_columns, vis_cat_columns)
                                           #transformed_test_errors_df)
