import copy

from torch import nn

from dataset import *
from learn_bot.engagement_aim.dataset import AimDataset
from learn_bot.libs.accuracy_and_loss import compute_loss, compute_accuracy, finish_accuracy, CUDA_DEVICE_STR, \
    CPU_DEVICE_STR
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes, PRIOR_TICKS, FUTURE_TICKS, CUR_TICK
from learn_bot.engagement_aim.lstm_aim_model import LSTMAimModel
from typing import List, Dict
from dataclasses import dataclass
import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm


@dataclass
class PolicyOutput:
    delta_view_angle_x: float
    delta_view_angle_y: float


# generate the input tensor for the next policy iteration
# create the dict for inserting a new training data point into the data frame
def get_x_field_str(tick: int = -1):
    if tick < 0:
        return f"delta view angle x (t-{abs(tick)})"
    else:
        return f"delta view angle x (t+{tick})"


def get_y_field_str(tick: int = -1):
    if tick < 0:
        return f"delta view angle y (t-{abs(tick)})"
    else:
        return f"delta view angle y (t+{tick})"


class PolicyHistory:
    row_dict: Dict
    input_tensor: torch.Tensor

    def __init__(self, row_dict: Dict, input_tensor: torch.Tensor):
        self.row_dict = row_dict
        self.input_tensor = input_tensor

    def add_row(self, policy_output: PolicyOutput, cts: IOColumnTransformers, new_row_dict: Dict,
                new_input_tensor: torch.Tensor, agg_dicts: List[Dict]):
        # update new input_tensor and row_dict by setting the view angles from old input_tensor
        # most recent values are form policy_output
        for i in range(PRIOR_TICKS, -1):
            new_row_dict[get_x_field_str(i)] = self.row_dict[get_x_field_str(i + 1)]
            new_row_dict[get_y_field_str(i)] = self.row_dict[get_y_field_str(i + 1)]

            cts.set_untransformed_output(new_input_tensor, get_x_field_str(i),
                                         cts.get_untransformed_output(self.input_tensor,
                                                                      get_x_field_str(i + 1)))
            cts.set_untransformed_output(new_input_tensor, get_y_field_str(i),
                                         cts.get_untransformed_output(self.input_tensor,
                                                                      get_y_field_str(i + 1)))

        new_row_dict[get_x_field_str()] = policy_output.delta_view_angle_x
        new_row_dict[get_y_field_str()] = policy_output.delta_view_angle_y
        cts.set_untransformed_output(new_input_tensor, get_x_field_str(), policy_output.delta_view_angle_x)
        cts.set_untransformed_output(new_input_tensor, get_y_field_str(), policy_output.delta_view_angle_y)

        self.row_dict = new_row_dict
        self.input_tensor = new_input_tensor
        agg_dicts.append(self.row_dict)


@dataclass
class RoundPolicyData:
    round_start_index: int
    round_end_index: int
    cur_index: int
    # this tracks history so it can produce inputs
    history_per_engagement: Dict[int, PolicyHistory]
    # this tracks last output, as output only used as input when hit next input
    last_output_per_engagement: Dict[int, PolicyOutput]
    # this tracks dicts (one dict per new row) for data points to add to data ste
    agg_dicts: Dict


def on_policy_inference(dataset: AimDataset, orig_df: pd.DataFrame, model: nn.Module,
                        cts: IOColumnTransformers) -> pd.DataFrame:
    agg_dicts = []
    model.eval()
    rounds_policy_data: Dict[int, RoundPolicyData] = {}
    for round_index, row in dataset.round_starts_ends.iterrows():
        rounds_policy_data[round_index] = RoundPolicyData(row['start index'], row['end index'], row['start index'],
                                                          {}, {}, {})
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
                        rounds_policy_data[valid_round_id].history_per_engagement[engagement_id].add_row(
                            rounds_policy_data[valid_round_id].last_output_per_engagement[engagement_id],
                            cts,
                            orig_df.loc[cur_index].to_dict(),
                            dataset[cur_index][0],
                            agg_dicts
                        )
                    else:
                        rounds_policy_data[valid_round_id].history_per_engagement[engagement_id] = PolicyHistory(
                            orig_df.loc[cur_index].to_dict(), dataset[cur_index][0])
                    round_row_tensors.append(rounds_policy_data[valid_round_id]
                                             .history_per_engagement[engagement_id].input_tensor)
                X_rolling = torch.stack(round_row_tensors, dim=0)
                pred = model(X_rolling.to(CUDA_DEVICE_STR)).to(CPU_DEVICE_STR).detach()
                #pred = model(X_rolling).detach()
                # need to add output to data set
                for i, valid_round_id in enumerate(valid_rounds):
                    cur_index = rounds_policy_data[valid_round_id].cur_index
                    engagement_id = dataset.engagement_id.loc[cur_index]
                    rounds_policy_data[valid_round_id].last_output_per_engagement[engagement_id] = PolicyOutput(
                        cts.get_untransformed_output(pred[i], "delta view angle x (t)"),
                        cts.get_untransformed_output(pred[i], "delta view angle y (t)")
                    )
                    rounds_policy_data[valid_round_id].cur_index += 1
                pbar.update(len(valid_rounds))

    # get last round worth of data
    agg_df = pd.DataFrame.from_dict(agg_dicts)
    return agg_df


