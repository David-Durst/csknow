from learn_bot.engagement_aim.column_management import PRIOR_TICKS, FUTURE_TICKS, PRIOR_TICKS_POS, CUR_TICK, \
    IOColumnTransformers
from learn_bot.engagement_aim.dad import get_x_field_str, get_y_field_str, on_policy_inference
from learn_bot.engagement_aim.train import train
from learn_bot.engagement_aim.vis import vis
import pandas as pd
from learn_bot.engagement_aim.dataset import data_path, AimDataset, input_column_types, output_column_types
from dataclasses import dataclass
from typing import Tuple, List, Dict
import copy
import numpy as np

from learn_bot.libs.df_grouping import get_row_as_dict_iloc


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class AimEngagementExample:
    mouse_xy: List[Point2D]


SEQUENCE_LENGTH = 50 + FUTURE_TICKS + PRIOR_TICKS_POS
straight_line_example = AimEngagementExample([Point2D(0., i * -0.1) for i in range(SEQUENCE_LENGTH)])
engagement_examples = [straight_line_example]


def build_aim_df(example_row_df: Dict) -> pd.DataFrame:
    result_dicts = []
    # 0 everything in dict
    for k in example_row_df.keys():
        if isinstance(example_row_df[k], int):
            example_row_df[k] = 0
        else:
            example_row_df[k] = 0.
    tick_id = 0
    for engagement_id in range(len(engagement_examples)):
        engagement_dicts = []
        for tick_in_engagement in range(len(engagement_examples[engagement_id].mouse_xy)):
            new_dict = copy.deepcopy(example_row_df)
            new_dict['engagement id'] = engagement_id
            new_dict['id'] = tick_id
            new_dict['tick id'] = tick_id
            new_dict['demo tick id'] = tick_id
            new_dict['game tick id'] = tick_id + PRIOR_TICKS
            new_dict['delta view angle x (t)'] = engagement_examples[engagement_id].mouse_xy[tick_in_engagement].x
            new_dict['delta view angle y (t)'] = engagement_examples[engagement_id].mouse_xy[tick_in_engagement].y
            for time_offset in range(PRIOR_TICKS, CUR_TICK + FUTURE_TICKS):
                if time_offset == 0:
                    continue
                tick_with_time_offset = tick_in_engagement + time_offset
                if tick_with_time_offset < 0 or \
                        tick_with_time_offset >= len(engagement_examples[engagement_id].mouse_xy):
                    continue
                new_dict[get_x_field_str(time_offset)] = engagement_examples[engagement_id] \
                    .mouse_xy[tick_with_time_offset].x
                new_dict[get_y_field_str(time_offset)] = engagement_examples[engagement_id] \
                    .mouse_xy[tick_with_time_offset].y
            engagement_dicts.append(new_dict)
            tick_id += 1
        result_dicts.extend(engagement_dicts[PRIOR_TICKS_POS:-1 * FUTURE_TICKS])
    return pd.DataFrame(result_dicts)


def vis_train():
    all_data_df = pd.read_csv(data_path)
    all_data_df = all_data_df.sort_values(['engagement id', 'tick id'])
    example_row = get_row_as_dict_iloc(all_data_df, 0)
    simple_df = build_aim_df(example_row)
    # vis(simple_df)
    train_result = train(simple_df, 0, 500, False, False)
    simple_pred_df = on_policy_inference(train_result.train_dataset, simple_df,
                                         train_result.model, train_result.column_transformers,
                                         True)
    vis(simple_df, simple_pred_df)



if __name__ == "__main__":
    vis_train()
