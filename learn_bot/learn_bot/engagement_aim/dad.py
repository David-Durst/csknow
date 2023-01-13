import copy

import torch
from torch import nn

from dataset import *
from learn_bot.engagement_aim.dataset import AimDataset
from learn_bot.libs.accuracy_and_loss import compute_loss, compute_accuracy, finish_accuracy, CUDA_DEVICE_STR, \
    CPU_DEVICE_STR
from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, ColumnTypes, PRIOR_TICKS, FUTURE_TICKS, \
    CUR_TICK, ModelOutput
from learn_bot.engagement_aim.lstm_aim_model import LSTMAimModel
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import torch.multiprocessing as mp
from tqdm import tqdm

from learn_bot.libs.df_grouping import get_row_as_dict_loc

# generate the input tensor for the next policy iteration
# create the dict for inserting a new training data point into the data frame
def get_x_field_str(tick: int = -1):
    return get_temporal_field_str(base_changed_offset_coordinates.attacker_x_view_angle, tick)


def get_y_field_str(tick: int = -1):
    return get_temporal_field_str(base_changed_offset_coordinates.attacker_y_view_angle, tick)

def get_holding_attack_field_str(tick: int = -1):
    return get_temporal_field_str(base_holding_attack, tick)


class PolicyHistory:
    row_series: pd.Series
    input_tensor: torch.Tensor

    def __init__(self, row_series: pd.Series, input_tensor: torch.Tensor):
        self.row_series = row_series
        self.input_tensor = input_tensor

    # for moving to next tick
    def add_row(self, cts: IOColumnTransformers, new_row_series: pd.Series, new_input_tensor: torch.Tensor, on_policy):
        # update new input_tensor and row_series by setting the view angles from old input_tensor
        # most recent values are form policy_output
        # -1 value is set to last preseriesion since finish_row updates self.row_series on last tick
        if on_policy:
            for i in range(PRIOR_TICKS, -1):
                new_row_series[get_x_field_str(i)] = self.row_series[get_x_field_str(i + 1)]
                new_row_series[get_y_field_str(i)] = self.row_series[get_y_field_str(i + 1)]

                cts.set_untransformed_input_value(new_input_tensor, get_x_field_str(i),
                                                  cts.get_untransformed_value(self.input_tensor,
                                                                              get_x_field_str(i + 1), True))
                cts.set_untransformed_input_value(new_input_tensor, get_y_field_str(i),
                                                  cts.get_untransformed_value(self.input_tensor,
                                                                              get_y_field_str(i + 1), True))

                x_series = new_row_series[get_x_field_str(i)]
                y_series = new_row_series[get_y_field_str(i)]
                x_tensor = cts.get_untransformed_value(new_input_tensor, get_x_field_str(i), True)
                y_tensor = cts.get_untransformed_value(new_input_tensor, get_y_field_str(i), True)
                if abs(x_series - x_tensor) > 0.0001:
                    print("x bad")
                if abs(y_series - y_tensor) > 0.0001:
                    print("y bad")
                #print(f"({x_series}, {y_series}), ({x_tensor},{y_tensor})")

        # last t output is new t-1 input
        new_row_series[get_x_field_str(-1)] = self.row_series[get_x_field_str(0)]
        new_row_series[get_y_field_str(-1)] = self.row_series[get_y_field_str(0)]
        cts.set_untransformed_input_value(new_input_tensor, get_x_field_str(-1), self.row_series[get_x_field_str(0)])
        cts.set_untransformed_input_value(new_input_tensor, get_y_field_str(-1), self.row_series[get_y_field_str(0)])

        self.row_series = new_row_series
        self.input_tensor = new_input_tensor

    # for finishing cur tick
    def finish_row(self, pred: ModelOutput, cts: IOColumnTransformers, agg_series: List[pd.Series],
                   result_str: Optional[List[str]] = None):

        # finish cur input_tensor by setting all the outputs
        # TODO: handle outputs other than aim
        for i in range(0, CUR_TICK + FUTURE_TICKS):
            self.row_series[get_x_field_str(i)] = cts.get_untransformed_value(pred, get_x_field_str(i), False)
            self.row_series[get_y_field_str(i)] = cts.get_untransformed_value(pred, get_y_field_str(i), False)
            self.row_series[get_holding_attack_field_str(i)] = cts.get_untransformed_value(pred, get_holding_attack_field_str(i), False)
            if result_str is not None:
                result_str.append(f"{i}: ({self.row_series[get_x_field_str(i)]:.2E},"
                                  f" {self.row_series[get_y_field_str(i)]:.2e}); ")

        if result_str is not None:
            result_str.append("\n")

        agg_series.append(self.row_series)


@dataclass
class RoundPolicyData:
    round_start_index: int
    round_end_index: int
    cur_index: int
    # this tracks history so it can produce inputs and save outputs
    history_per_engagement: Dict[int, PolicyHistory]


#global_model = None
#global_model = torch.jit.load(manual_data_path.parent / 'reduced_weighted_firing_script_model.pt')

def on_policy_inference(dataset: AimDataset, orig_df: pd.DataFrame, model: nn.Module,
                        cts: IOColumnTransformers, on_policy=True) -> pd.DataFrame:
    agg_series = []
    model.eval()
    result_strs = None #set to [] to get add_row printing
    rounds_policy_data: Dict[int, RoundPolicyData] = {}
    tmp_tick = 0
    for round_index, row in dataset.round_starts_ends.iterrows():
        rounds_policy_data[round_index] = RoundPolicyData(row['start index'], row['end index'], row['start index'], {})
    with open(manual_data_path.parent / 'dad_output2.csv', 'w+') as f:
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
                                orig_df.loc[cur_index].copy(),
                                dataset[cur_index][0],
                                on_policy
                            )
                        else:
                            rounds_policy_data[valid_round_id].history_per_engagement[engagement_id] = PolicyHistory(
                                orig_df.loc[cur_index].copy(), dataset[cur_index][0])
                        round_row_tensors.append(rounds_policy_data[valid_round_id]
                                                 .history_per_engagement[engagement_id].input_tensor)
                    #tmp_vals = [0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,0,0,100,100,1,1,1,1440.06,1112.91,55.2335,1417.53,1653.84,56.6999,0,0,0,0,0,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,92.4194,88.5806,0,-1.74763,1.77859,0,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,-0.302368,-2.20575,-0,-8.99872,-1.1727,-1.90338,0,0,0,0,0,0,0,0,0,0,0,0,0,3]
                    #tmp_tensor = torch.tensor(tmp_vals)
                    #if False and tmp_tick == 0:
                        #print((tmp_tensor - round_row_tensors[0]).tolist())
                        #X_rolling = torch.stack(round_row_tensors + [tmp_tensor], dim=0)
                    X_rolling = torch.stack(round_row_tensors, dim=0)
                    pred = model(X_rolling.to(CUDA_DEVICE_STR))
                    pred = (pred[0].to(CPU_DEVICE_STR).detach(), pred[1].to(CPU_DEVICE_STR).detach())
                    #print(",".join([str(x.item()) for x in list(X_rolling[0])]))
                    if False and tmp_tick == 0:
                        #pred_loaded = global_model(X_rolling)
                        #pred_loaded = (pred_loaded[0].to(CPU_DEVICE_STR).detach(), pred_loaded[1].to(CPU_DEVICE_STR).detach())
                        tmp_dict = cts.get_untransformed_values(X_rolling[0], True)
                        base_keys = ["hit victim","recoil index","ticks since last fire","ticks since last holding attack","victim visible","victim visible yet","victim alive","attacker eye pos x","attacker eye pos y","attacker eye pos z","victim eye pos x","victim eye pos y","victim eye pos z","attacker vel x","attacker vel y","attacker vel z","victim vel x","victim vel y","victim vel z","ideal view angle x","ideal view angle y","delta relative first head view angle x","delta relative first head view angle y","scaled recoil angle x","scaled recoil angle y","victim relative first head min view angle x","victim relative first head min view angle y","victim relative first head max view angle x","victim relative first head max view angle y","victim relative first head cur head view angle x","victim relative first head cur head view angle y","holding attack"]
                        extra_keys = ["attacker view angle x","attacker view angle y"]
                        print(",".join(base_keys + extra_keys), file=f)
                        for i in range(PRIOR_TICKS,0):
                            for k in base_keys:
                                print(f"{tmp_dict[get_temporal_field_str(k, i)]},", end="", file=f)
                            for k in extra_keys:
                                print(f"{orig_df.loc[cur_index, get_temporal_field_str(k, i)]},", end="", file=f)
                            print("", file=f)
                        print(cts.input_types.column_names(), file=f)
                        print(round_row_tensors[0].tolist(), file=f)
                        print("", file=f)
                        #zx = [(k, tmp_dict[k]) for k in tmp_dict if 'delta relative first head view angle x' in k]
                        #zy = [(k, tmp_dict[k]) for k in tmp_dict if 'delta relative first head view angle y' in k]
                        #vy = [(k, tmp_dict[k]) for k in tmp_dict if 'victim eye pos y' in k]
                        #for k,v in zx:
                        #    print(f"{k}:{v}")
                        #for k,v in zy:
                        #    print(f"{k}:{v}")
                        #for k, v in vy:
                        #    print(f"{k}:{v}")
                        print(f"{pred[1][0,0].item()}, {pred[1][0,14].item()}", file=f)
                    tmp_tick += 1
                    #pred = model(X_rolling).detach()
                    # need to add output to data set
                    for i, valid_round_id in enumerate(valid_rounds):
                        cur_index = rounds_policy_data[valid_round_id].cur_index
                        engagement_id = dataset.engagement_id.loc[cur_index]
                        # save all predictions for output row
                        rounds_policy_data[valid_round_id].history_per_engagement[engagement_id] \
                            .finish_row((pred[0][i], pred[1][i]), cts, agg_series, result_strs)
                        rounds_policy_data[valid_round_id].cur_index += 1
                    pbar.update(len(valid_rounds))

    if result_strs is not None:
        print("".join(result_strs))
    # get last round worth of data
    agg_df = pd.DataFrame(agg_series)
    return agg_df


def create_dad_dataset(pred_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    column_names = pred_df.columns.tolist()
    pred_column_names = [name for name in column_names if "(t-" in name or name == "index"]
    pred_df_subset = pred_df.loc[:, pred_column_names]
    train_column_names = [name for name in column_names if "(t-" not in name or name == "index"]
    train_df_subset = train_df.loc[:, train_column_names]
    return train_df_subset.merge(pred_df_subset, on="index", how="inner")

