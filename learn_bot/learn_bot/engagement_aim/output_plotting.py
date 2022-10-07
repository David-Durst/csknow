import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt
from typing import Dict
from learn_bot.engagement_aim.accuracy_and_loss import CPU_DEVICE_STR

import torch

from learn_bot.engagement_aim.aim_model import AimModel
from learn_bot.engagement_aim.column_management import IOColumnTransformers

float_column_x_axes: List[str] = ['yaw degree', 'pitch degree', 'yaw degree', 'pitch degree',
                                  'yaw degree', 'pitch degree', 'hammer units']

cat_column_x_axes: List[str] = ['weapon type']

INCH_PER_FIG = 4


def plot_untransformed_and_transformed(title: str, cts: IOColumnTransformers, df, float_cols, cat_cols,
                                       transformed_df):
    # plot untransformed and transformed outputs
    fig = plt.figure(figsize=(INCH_PER_FIG * len(float_cols), 3.5 * INCH_PER_FIG), constrained_layout=True)
    fig.suptitle(title)
    num_rows = 2
    if cat_cols:
        num_rows += 1
    subfigs = fig.subfigures(nrows=num_rows, ncols=1)

    # untransformed
    axs = subfigs[0].subplots(1, len(float_cols), squeeze=False)
    subfigs[0].suptitle('float untransformed')
    axs[0][0].set_ylabel('num points')
    for i in range(len(float_cols)):
        df.hist(float_cols[i], ax=axs[0][i], bins=100)
        axs[0][i].set_xlabel(float_column_x_axes[i % len(float_column_x_axes)])

    # transformed
    axs = subfigs[1].subplots(1, len(float_cols), squeeze=False)
    subfigs[1].suptitle('float transformed')
    axs[0][0].set_ylabel('num points')
    for i in range(len(float_cols)):
        transformed_df.hist(float_cols[i], ax=axs[0][i], bins=100)
        axs[0][i].set_xlabel(float_column_x_axes[i % len(float_column_x_axes)] + ' standardized')
    # plt.tight_layout()

    # categorical
    if cat_cols:
        axs = subfigs[2].subplots(1, len(cat_cols), squeeze=False)
        subfigs[2].suptitle('categorical')
        axs[0][0].set_ylabel('num points')
        for i in range(len(cat_cols)):
            axs[0][i].set_xlabel(cat_column_x_axes[i % len(cat_column_x_axes)])
            df.loc[:, cat_cols[i]].value_counts().plot.bar(ax=axs[0][0])
    plt.show()


class ModelOutputRecording:
    train_outputs_untransformed: torch.Tensor
    train_outputs_transformed: torch.Tensor
    test_outputs_untransformed: torch.Tensor
    test_outputs_transformed: torch.Tensor
    test_errors_untransformed: torch.Tensor
    test_errors_transformed: torch.Tensor
    model: AimModel

    def __init__(self, model: AimModel):
        self.train_outputs_untransformed = None
        self.train_outputs_transformed = None
        self.test_outputs_untransformed = None
        self.test_outputs_transformed = None
        self.test_errors_untransformed = None
        self.test_errors_transformed = None
        self.model = model

    def record_output(self, pred: torch.Tensor, Y: torch.Tensor, transformed_Y: torch.Tensor, train):
        pred = pred.detach()
        Y = Y.detach()
        transformed_Y = transformed_Y.detach()

        if train:
            if self.train_outputs_untransformed is None:
                self.train_outputs_untransformed = self.model.get_untransformed_outputs(pred)
                self.train_outputs_transformed = self.model.get_transformed_outputs(pred)
            else:
                self.train_outputs_untransformed = torch.cat(
                    [self.train_outputs_untransformed, self.model.get_untransformed_outputs(pred)], 0)
                self.train_outputs_transformed = torch.cat(
                    [self.train_outputs_untransformed, self.model.get_transformed_outputs(pred)], 0)
        else:
            if self.test_outputs_untransformed is None:
                self.test_outputs_untransformed = self.model.get_untransformed_outputs(pred)
                self.test_outputs_transformed = self.model.get_transformed_outputs(pred)
                self.test_errors_untransformed = self.model.get_untransformed_outputs(pred) - Y
                self.test_errors_transformed = self.model.get_transformed_outputs(pred) - transformed_Y
            else:
                self.test_outputs_untransformed = torch.cat(
                    [self.test_outputs_untransformed, self.model.get_untransformed_outputs(pred)], 0)
                self.test_outputs_transformed = torch.cat(
                    [self.test_outputs_transformed, self.model.get_transformed_outputs(pred)], 0)
                self.test_errors_untransformed = torch.cat(
                    [self.test_errors_untransformed, self.model.get_untransformed_outputs(pred) - Y], 0)
                self.test_errors_transformed = torch.cat(
                    [self.test_errors_transformed, self.model.get_transformed_outputs(pred) - transformed_Y], 0)

    def plot(self, cts: IOColumnTransformers, vis_float_columns: List[str]):
        test_df = pd.DataFrame(self.test_outputs_untransformed.to(CPU_DEVICE_STR),
                               columns=cts.output_types.column_names())
        transformed_test_df = pd.DataFrame(self.test_outputs_transformed.to(CPU_DEVICE_STR),
                                           columns=cts.output_types.column_names())
        plot_untransformed_and_transformed('test predictions', cts, test_df, vis_float_columns, [],
                                           transformed_test_df)

        test_errors_df = pd.DataFrame(self.test_errors_untransformed.to(CPU_DEVICE_STR),
                                      columns=cts.output_types.column_names())
        transformed_test_errors_df = pd.DataFrame(self.test_errors_transformed.to(CPU_DEVICE_STR),
                                                  columns=cts.output_types.column_names())
        plot_untransformed_and_transformed('test errors', cts, test_errors_df, vis_float_columns, [],
                                           transformed_test_errors_df)
