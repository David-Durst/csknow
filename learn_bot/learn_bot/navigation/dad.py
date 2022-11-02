import copy

from dataset import *
from learn_bot.engagement_aim.dataset import AimDataset
from learn_bot.libs.accuracy_and_loss import compute_loss, compute_accuracy, finish_accuracy, CUDA_DEVICE_STR, \
    CPU_DEVICE_STR
from learn_bot.engagement_aim.column_management import IOColumnTransformers, ColumnTypes, PRIOR_TICKS, FUTURE_TICKS, CUR_TICK
from learn_bot.engagement_aim.lstm_aim_model import LSTMAimModel
from typing import List, Dict
from dataclasses import dataclass
from alive_progress import alive_bar
import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm


@dataclass
class PolicyOutput:
    delta_view_angle_x: float
    delta_view_angle_y: float


class PolicyHistory:
    # generate the input tensor for the next policy iteration
    # create the dict for inserting a new training data point into the data frame
    def get_x_field_str(self, tick: int = -1):
        return f"delta view angle x (t-{abs(tick)})"

    def get_y_field_str(self, tick: int = -1):
        return f"delta view angle y (t-{abs(tick)})"

    row_dict: Dict
    input_tensor: torch.Tensor

    def __init__(self, row_dict: Dict, input_tensor: torch.Tensor):
        self.row_dict = row_dict
        self.input_tensor = input_tensor

    def add_row(self, policy_output: PolicyOutput, model: LSTMAimModel, new_row_dict: Dict,
                new_input_tensor: torch.Tensor, agg_dicts: List[Dict]):
        # update new input_tensor and row_dict by setting the view angles from old input_tensor
        # most recent values are form policy_output
        for i in range(PRIOR_TICKS, -1):
            new_row_dict[self.get_x_field_str(i)] = self.row_dict[self.get_x_field_str(i + 1)]
            new_row_dict[self.get_y_field_str(i)] = self.row_dict[self.get_y_field_str(i + 1)]

            model.set_untransformed_output(new_input_tensor, self.get_x_field_str(i),
                                           model.get_untransformed_output(self.input_tensor,
                                                                          self.get_x_field_str(i + 1)))
            model.set_untransformed_output(new_input_tensor, self.get_y_field_str(i),
                                           model.get_untransformed_output(self.input_tensor,
                                                                          self.get_y_field_str(i + 1)))

        new_row_dict[self.get_x_field_str()] = policy_output.delta_view_angle_x
        new_row_dict[self.get_y_field_str()] = policy_output.delta_view_angle_y
        model.set_untransformed_output(new_input_tensor, self.get_x_field_str(), policy_output.delta_view_angle_x)
        model.set_untransformed_output(new_input_tensor, self.get_y_field_str(), policy_output.delta_view_angle_y)

        self.row_dict = new_row_dict
        self.input_tensor = new_input_tensor
        agg_dicts.append(self.row_dict)


def on_policy_inference(dataset: AimDataset, orig_df: pd.DataFrame, model: LSTMAimModel, rounds, pid,
                        lock, return_dict) -> pd.DataFrame:
    agg_dicts = []
    inner_agg_df = pd.DataFrame()
    model.eval()
    prior_row_round_id = -1
    # this tracks history so it can produce inputs
    history_per_engagement: Dict[int, PolicyHistory] = {}
    # this tracks last output, as output only used as input when hit next input
    last_output_per_engagement: Dict[int, PolicyOutput] = {}
    num_rounds = dataset.round_id.isin(rounds).sum()
    torch.set_num_threads(1)
    with torch.no_grad():
        with tqdm(total=num_rounds, disable=False, position=pid) as pbar:
            for i in range(len(dataset)):
                if dataset.round_id.iloc[i] not in rounds:
                    continue
                if prior_row_round_id != dataset.round_id.iloc[i]:
                    round_df = pd.DataFrame.from_dict(agg_dicts)
                    inner_agg_df = pd.concat([inner_agg_df, round_df], ignore_index=True)
                    agg_dicts = []
                    history_per_engagement = {}
                    last_output_per_engagement = {}
                    if i != 0:
                        break
                prior_row_round_id = dataset.round_id.iloc[i]
                engagement_id = dataset.engagement_id.iloc[i]
                if engagement_id in history_per_engagement:
                    history_per_engagement[engagement_id].add_row(
                        last_output_per_engagement[engagement_id],
                        model,
                        orig_df.iloc[i].to_dict(),
                        torch.unsqueeze(dataset[i][0], dim=0).detach(),
                        agg_dicts
                    )
                else:
                    history_per_engagement[engagement_id] = PolicyHistory(
                        orig_df.iloc[i].to_dict(), torch.unsqueeze(dataset[i][0], dim=0).detach())
                X_rolling = history_per_engagement[engagement_id].input_tensor
                #pred = model(X_rolling.to(CUDA_DEVICE_STR)).to(CPU_DEVICE_STR).detach()
                pred = model(X_rolling).detach()
                # need to add output to data set
                last_output_per_engagement[engagement_id] = PolicyOutput(
                    model.get_untransformed_output(pred, "delta view angle x (t)"),
                    model.get_untransformed_output(pred, "delta view angle y (t)")
                )
                with lock:
                    pbar.update(1)

    # get last round worth of data
    round_df = pd.DataFrame.from_dict(agg_dicts)
    inner_agg_df = pd.concat([inner_agg_df, round_df], ignore_index=True)

    return_dict[pid] = inner_agg_df

