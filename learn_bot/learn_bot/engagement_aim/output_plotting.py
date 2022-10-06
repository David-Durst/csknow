import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt
from typing import Dict
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
        axs[0][i].set_xlabel(float_column_x_axes[i])

    # transformed
    axs = subfigs[1].subplots(1, len(float_cols), squeeze=False)
    subfigs[1].suptitle('float transformed')
    axs[0][0].set_ylabel('num points')
    for i in range(len(float_cols)):
        transformed_df.hist(float_cols[i], ax=axs[0][i], bins=100)
        axs[0][i].set_xlabel(float_column_x_axes[i])
    # plt.tight_layout()

    # categorical
    if cat_cols:
        axs = subfigs[2].subplots(1, len(cat_cols), squeeze=False)
        subfigs[2].suptitle('categorical')
        axs[0][0].set_ylabel('num points')
        for i in range(len(cat_cols)):
            axs[0][i].set_xlabel(cat_column_x_axes[i])
            df.loc[:, cat_cols[i]].value_counts().plot.bar(ax=axs[0][0])
    plt.show()


class ModelOutputRecording:
    train_outputs_untransformed: Dict[str, List[float]]
    train_outputs_transformed: Dict[str, List[float]]
    test_outputs_untransformed: Dict[str, List[float]]
    test_outputs_transformed: Dict[str, List[float]]
    test_errors_untransformed: Dict[str, List[float]]
    test_errors_transformed: Dict[str, List[float]]

    def __init__(self, cts: IOColumnTransformers):
        self.train_outputs_untransformed = {}
        self.train_outputs_transformed = {}
        self.test_outputs_untransformed = {}
        self.test_outputs_transformed = {}
        self.test_errors_untransformed = {}
        self.test_errors_transformed = {}

        for name in cts.output_types.column_names():
            self.train_outputs_untransformed[name] = []
            self.train_outputs_transformed[name] = []
            self.test_outputs_untransformed[name] = []
            self.test_outputs_transformed[name] = []
            self.test_errors_untransformed[name] = []
            self.test_errors_transformed[name] = []

    def record_output(self, cts: IOColumnTransformers, pred, Y, transformed_Y, train):
        column_names = cts.output_types.column_names()
        for name, r in zip(column_names, cts.get_name_ranges(False)):
            # compute accuracy using unnormalized outputs on end
            untransformed_r = range(r.start + len(column_names), r.stop + len(column_names))
            if train:
                self.train_outputs_untransformed[name].extend(pred[:, untransformed_r].reshape(-1).tolist())
                self.train_outputs_transformed[name].extend(pred[:, r].reshape(-1).tolist())
            else:
                self.test_outputs_untransformed[name].extend(pred[:, untransformed_r].reshape(-1).tolist())
                self.test_outputs_transformed[name].extend(pred[:, r].reshape(-1).tolist())
                self.test_errors_untransformed[name].extend((pred[:, untransformed_r] - Y[:, r]).reshape(-1).tolist())
                self.test_errors_transformed[name] \
                    .extend((pred[:, r] - transformed_Y[:, r]).reshape(-1).tolist())

    def plot(self, cts: IOColumnTransformers, float_cols):
        test_df = pd.DataFrame.from_dict(self.test_outputs_untransformed)
        transformed_test_df = pd.DataFrame.from_dict(self.test_outputs_transformed)
        plot_untransformed_and_transformed('test predictions', cts, test_df, float_cols, [], transformed_test_df)

        test_errors_df = pd.DataFrame.from_dict(self.test_errors_untransformed)
        transformed_test_errors_df = pd.DataFrame.from_dict(self.test_errors_transformed)
        plot_untransformed_and_transformed('test errors', cts, test_errors_df, float_cols, [], transformed_test_errors_df)
