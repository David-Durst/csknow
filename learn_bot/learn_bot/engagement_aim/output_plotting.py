import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt
from typing import Dict
from learn_bot.engagement_aim.column_management import IOColumnTransformers


def plot_untransformed_and_transformed(title: str, cts: IOColumnTransformers, df, transformed_df=None):
    # plot untransformed and transformed outputs
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(title)
    subfigs = fig.subfigures(nrows=2, ncols=1)

    # untransformed
    axs = subfigs[0].subplots(1, 2)
    subfigs[0].suptitle('untransformed')
    df.hist('delta view angle x (t - 0)', ax=axs[0], bins=100)
    axs[0].set_xlabel('yaw degree')
    axs[0].set_ylabel('num points')
    df.hist('delta view angle y (t - 0)', ax=axs[1], bins=100)
    axs[1].set_xlabel('pitch degree')

    # transformed
    axs = subfigs[1].subplots(1, 2)
    subfigs[1].suptitle('transformed')
    if transformed_df is None:
        transformed_df = pd.DataFrame(
            cts.output_ct.transform(df.loc[:, cts.output_types.column_names()]),
            columns=cts.output_types.column_names())
    transformed_df.hist('delta view angle x (t - 0)', ax=axs[0], bins=100)
    axs[0].set_xlabel('standardized yaw degree')
    axs[0].set_ylabel('num points')
    transformed_df.hist('delta view angle y (t - 0)', ax=axs[1], bins=100)
    axs[1].set_xlabel('standardized pitch degree')
    # plt.tight_layout()
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

    def plot(self, cts: IOColumnTransformers):
        test_df = pd.DataFrame.from_dict(self.test_outputs_untransformed)
        transformed_test_df = pd.DataFrame.from_dict(self.test_outputs_transformed)
        plot_untransformed_and_transformed('test predictions', cts, test_df, transformed_test_df)

        test_errors_df = pd.DataFrame.from_dict(self.test_errors_untransformed)
        transformed_test_errors_df = pd.DataFrame.from_dict(self.test_errors_transformed)
        plot_untransformed_and_transformed('test errors', cts, test_errors_df, transformed_test_errors_df)
