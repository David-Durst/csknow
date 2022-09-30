import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt
from typing import Dict
from learn_bot.engagement_aim.column_management import IOColumnTransformers

def plot_untransformed_and_transformed(cts: IOColumnTransformers, df, transformed_df = None, labels: bool = True):
    # plot untransformed and transformed outputs
    fig = plt.figure(constrained_layout=True)
    if labels:
        fig.suptitle('train+test labels')
    else:
        fig.suptitle('test predictions')
    subfigs = fig.subfigures(nrows=2, ncols=1)

    # untransformed
    axs = subfigs[0].subplots(1,2)
    subfigs[0].suptitle('untransformed')
    df.hist('delta view angle x (t - 0)', ax=axs[0], bins=100)
    axs[0].set_xlabel('yaw degree')
    axs[0].set_ylabel('num points')
    df.hist('delta view angle y (t - 0)', ax=axs[1], bins=100)
    axs[1].set_xlabel('pitch degree')

    # transformed
    axs = subfigs[1].subplots(1,2)
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
    #plt.tight_layout()
    plt.show()


class ModelOutputRecording:
    train_outputs_untransformed: Dict[str, List[float]]
    train_outputs_transformed: Dict[str, List[float]]
    test_outputs_untransformed: Dict[str, List[float]]
    test_outputs_transformed: Dict[str, List[float]]

    def __init__(self, cts: IOColumnTransformers):
        for name in cts.output_types.column_names(False):
            self.train_outputs_untransformed[name] = []
            self.train_outputs_transformed[name] = []
            self.test_outputs_untransformed[name] = []
            self.test_outputs_transformed[name] = []

    def record_output(self, cts: IOColumnTransformers, pred, train):
        column_names = cts.output_types.column_names()
        for name, unadjusted_r in zip(column_names, cts.get_name_ranges(False)):
            # compute accuracy using unnormalized outputs on end
            r = range(unadjusted_r.start + len(column_names), unadjusted_r.stop + len(column_names))
            if train:
                self.train_outputs_untransformed[name].append(list(pred[:, r]))
                self.train_outputs_transformed[name].append(list(pred[:, unadjusted_r]))
            else:
                self.test_outputs_untransformed[name].append(list(pred[:, r]))
                self.test_outputs_transformed[name].append(list(pred[:, unadjusted_r]))

    def plot(self, cts: IOColumnTransformers):
        df = pd.DataFrame.from_dict(self.test_outputs_untransformed)
        transformed_df = pd.DataFrame.from_dict(self.test_outputs_transformed)
        plot_untransformed_and_transformed(cts, df, transformed_df, False)

