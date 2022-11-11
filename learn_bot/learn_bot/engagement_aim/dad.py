import copy

import torch
from torch import nn

from dataset import *
from learn_bot.engagement_aim.dataset import AimDataset
from learn_bot.libs.accuracy_and_loss import compute_loss, compute_accuracy, finish_accuracy, CUDA_DEVICE_STR, \
    CPU_DEVICE_STR
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes, PRIOR_TICKS, FUTURE_TICKS, CUR_TICK
from learn_bot.engagement_aim.lstm_aim_model import LSTMAimModel
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm

from learn_bot.libs.df_grouping import get_row_as_dict_loc


@dataclass
class PolicyOutput:
    delta_view_angle_x: float
    delta_view_angle_y: float


# generate the input tensor for the next policy iteration
# create the dict for inserting a new training data point into the data frame
def get_x_field_str(tick: int = -1):
    if tick < 0:
        return f"delta view angle x (t-{abs(tick)})"
    elif tick == 0:
        return f"delta view angle x (t)"
    else:
        return f"delta view angle x (t+{tick})"


def get_y_field_str(tick: int = -1):
    if tick < 0:
        return f"delta view angle y (t-{abs(tick)})"
    elif tick == 0:
        return f"delta view angle y (t)"
    else:
        return f"delta view angle y (t+{tick})"


class PolicyHistory:
    row_dict: Dict
    input_tensor: torch.Tensor

    def __init__(self, row_dict: Dict, input_tensor: torch.Tensor):
        self.row_dict = row_dict
        self.input_tensor = input_tensor

    # for moving to next tick
    def add_row(self, cts: IOColumnTransformers, new_row_dict: Dict, new_input_tensor: torch.Tensor):
        # update new input_tensor and row_dict by setting the view angles from old input_tensor
        # most recent values are form policy_output
        # -1 value is set to last prediction since finish_row updates self.row_dict on last tick
        for i in range(PRIOR_TICKS, 0):
            new_row_dict[get_x_field_str(i)] = self.row_dict[get_x_field_str(i + 1)]
            new_row_dict[get_y_field_str(i)] = self.row_dict[get_y_field_str(i + 1)]

            cts.set_untransformed_output(new_input_tensor, get_x_field_str(i),
                                         cts.get_untransformed_output(self.input_tensor,
                                                                      get_x_field_str(i + 1)))
            cts.set_untransformed_output(new_input_tensor, get_y_field_str(i),
                                         cts.get_untransformed_output(self.input_tensor,
                                                                      get_y_field_str(i + 1)))

        self.row_dict = new_row_dict
        self.input_tensor = new_input_tensor

    # for finishing cur tick
    def finish_row(self, pred: torch.Tensor, cts: IOColumnTransformers, agg_dicts: List[Dict],
                   result_str: Optional[List[str]] = None):

        # finish cur input_tensor by setting all the outputs
        # TODO: handle outputs other than aim
        for i in range(0, CUR_TICK + FUTURE_TICKS):
            self.row_dict[get_x_field_str(i)] = cts.get_untransformed_output(pred, get_x_field_str(i))
            self.row_dict[get_y_field_str(i)] = cts.get_untransformed_output(pred, get_y_field_str(i))
            if result_str is not None:
                result_str.append(f"{i}: ({self.row_dict[get_x_field_str(i)]:.2E},"
                                  f" {self.row_dict[get_y_field_str(i)]:.2e}); ")

        if result_str is not None:
            result_str.append("\n")

        agg_dicts.append(self.row_dict)


@dataclass
class RoundPolicyData:
    round_start_index: int
    round_end_index: int
    cur_index: int
    # this tracks history so it can produce inputs and save outputs
    history_per_engagement: Dict[int, PolicyHistory]


def on_policy_inference(dataset: AimDataset, orig_df: pd.DataFrame, model: nn.Module,
                        cts: IOColumnTransformers) -> pd.DataFrame:
    agg_dicts = []
    model.eval()
    result_strs = []
    rounds_policy_data: Dict[int, RoundPolicyData] = {}
    for round_index, row in dataset.round_starts_ends.iterrows():
        rounds_policy_data[round_index] = RoundPolicyData(row['start index'], row['end index'], row['start index'], {})
    with torch.no_grad():
        with tqdm(total=len(dataset), disable=False) as pbar:
            while True:
                # collect valid rounds, end if no rounds left to analyze
                valid_rounds = []
                for round_id, round_policy_data in rounds_policy_data.items():
                    if round_policy_data.cur_index <= round_policy_data.round_end_index:
                        valid_rounds.append(round_id)
                if len(valid_rounds) == 0:
                    break

                round_row_tensors = []
                for valid_round_id in valid_rounds:
                    cur_index = rounds_policy_data[valid_round_id].cur_index
                    engagement_id = dataset.engagement_id.loc[cur_index]
                    if engagement_id in rounds_policy_data[valid_round_id].history_per_engagement:
                        # want to take most of real data (like recoil) and just shift in old predictions about
                        # mouse x and y
                        rounds_policy_data[valid_round_id].history_per_engagement[engagement_id].add_row(
                            cts,
                            get_row_as_dict_loc(orig_df, cur_index),
                            dataset[cur_index][0],
                        )
                    else:
                        rounds_policy_data[valid_round_id].history_per_engagement[engagement_id] = PolicyHistory(
                            get_row_as_dict_loc(orig_df, cur_index), dataset[cur_index][0])
                    round_row_tensors.append(rounds_policy_data[valid_round_id]
                                             .history_per_engagement[engagement_id].input_tensor)
                X_rolling = torch.stack(round_row_tensors, dim=0)
                pred = model(X_rolling.to(CUDA_DEVICE_STR)).to(CPU_DEVICE_STR).detach()
                #pred = model(X_rolling).detach()
                # need to add output to data set
                for i, valid_round_id in enumerate(valid_rounds):
                    cur_index = rounds_policy_data[valid_round_id].cur_index
                    engagement_id = dataset.engagement_id.loc[cur_index]
                    # save all predictions for output row
                    rounds_policy_data[valid_round_id].history_per_engagement[engagement_id].finish_row(pred[i], cts,
                                                                                                        agg_dicts,
                                                                                                        result_strs)
                    rounds_policy_data[valid_round_id].cur_index += 1
                pbar.update(len(valid_rounds))

    print("".join(result_strs))
    # get last round worth of data
    agg_df = pd.DataFrame.from_dict(agg_dicts)
    return agg_df


