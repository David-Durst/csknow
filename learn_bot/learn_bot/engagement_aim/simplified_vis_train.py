from learn_bot.engagement_aim.train import train
from learn_bot.engagement_aim.vis import vis
import pandas as pd
from learn_bot.engagement_aim.dataset import data_path
from dataclasses import dataclass
from typing import Tuple, List, Dict
import copy


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class AimEngagementExample:
    mouse_xy: List[Point2D]


straight_line_example = AimEngagementExample([Point2D(0., i * 0.1) for i in range(11)])
engagement_examples = [straight_line_example]


def build_aim_df(example_row_df: Dict) -> pd.DataFrame:
    result_dicts = []
    # 0 everything in dict
    for k in example_row_df.keys():
        example_row_df[k] = 0.
    tick_id = 0
    for engagement_id in range(len(engagement_examples)):
        for tick_in_engagement in range(len(engagement_examples[engagement_id].mouse_xy)):
            new_dict = copy.deepcopy(example_row_df)
            new_dict['id'] = tick_id
            new_dict['tick id'] = tick_id
            new_dict['game tick id'] = tick_id
            new_dict['demo tick id'] = tick_id
            new_dict['delta view angle x (t)'] = engagement_examples[engagement_id].mouse_xy[tick_in_engagement].x
            new_dict['delta view angle y (t)'] = engagement_examples[engagement_id].mouse_xy[tick_in_engagement].y
            result_dicts.append(new_dict)
    return pd.DataFrame(result_dicts)


def vis_train():
    all_data_df = pd.read_csv(data_path)
    all_data_df = all_data_df.sort_values(['engagement id', 'tick id'])
    example_row = all_data_df.iloc[0, :].to_dict()
    simple_df = build_aim_df(example_row)
    vis(simple_df)
    train(simple_df, 0, False)


if __name__ == "__main__":
    vis_train()
