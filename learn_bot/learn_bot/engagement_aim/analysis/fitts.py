from typing import Dict, List

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from math import pow, sqrt, fabs, log2
import copy

from matplotlib.figure import Figure

bot_timing_path = Path(__file__).parent / "test_timing_data" / "bot" / "test_timing.csv"
bot_event_timing_path = Path(__file__).parent / "test_timing_data" / "bot" / "test_event_timing.csv"
human_timing_path = Path(__file__).parent / "test_timing_data" / "human" / "test_timing.csv"
human_event_timing_path = Path(__file__).parent / "test_timing_data" / "human" / "test_event_timing.csv"
plot_path = Path(__file__).parent / "test_timing_data" / "timing_result.png"
success_path = Path(__file__).parent / "test_timing_data" / "success.csv"

# use ramp so can isolate left/right movement coming right after ramp from left/right looking that comes later
# didn't use consistent naming, so need to check two sets of options
movement_options_a = ["CloseNone", "CloseForward", "CloseLeft", "CloseRight"]
movement_options_b = ["RampNone", "RampForward", "RampLeft", "RampRight"]
movement_names = ["Still", "Forward", "Left", "Right"]

none_red = (1., 0., 0., 1.)
forward_green = (0., 1., 0., 1.)
left_blue = (0., 0., 1., 1.)
right_purple = (1., 0., 1., 1.)
movement_name_to_color = {
    movement_names[0]: none_red,
    movement_names[1]: forward_green,
    movement_names[2]: left_blue,
    movement_names[3]: right_purple,
}

# column names
test_name_col = "test name"
test_id_col = "test id"
end_time_col = "end time"
success_col = "success"
event_name_col = "event name"
event_id_col = "event id"
event_time_col = "event time"
payload_col = "payload"

# event names
weapon_fire_event = 'weapon fire'
hurt_event = 'hurt'
angular_distance_x_event = 'angular distance x'
angular_distance_y_event = 'angular distance y'
angular_size_min_x_event = 'angular target min x'
angular_size_min_y_event = 'angular target min y'
angular_size_max_x_event = 'angular target max x'
angular_size_max_y_event = 'angular target max y'

INVALID_TIME = -1.

# derived column names
movement_name_col = "Movement"
index_of_difficulty_col = "Index of Difficulty (ID, Bits)"
first_shot_col = "First Shot Time (s)"
first_hit_col = "First Hit Time (s)"

class TestData:
    test_name: str
    movement_name: str
    angular_distance_x: float
    angular_distance_y: float
    angular_size_min_x: float
    angular_size_min_y: float
    angular_size_max_x: float
    angular_size_max_y: float

    angular_distance: float
    angular_width: float
    index_of_difficulty: float

    success: bool
    first_shot: float
    first_hit: float
    end_time: float

    def __init__(self, test_name: str, movement_name: str,
                 angular_distance_x: float, angular_distance_y: float,
                 angular_size_min_x: float, angular_size_min_y: float,
                 angular_size_max_x: float, angular_size_max_y: float,
                 success: bool, first_shot: float, first_hit: float, end_time: float):
        self.test_name = test_name
        self.movement_name = movement_name
        self.angular_distance_x = angular_distance_x
        self.angular_distance_y = angular_distance_y
        self.angular_size_min_x = angular_size_min_x
        self.angular_size_min_y = angular_size_min_y
        self.angular_size_max_x = angular_size_max_x
        self.angular_size_max_y = angular_size_max_y

        self.angular_distance = sqrt(pow(angular_distance_x, 2.) + pow(angular_distance_y, 2.))
        self.angular_width = fabs(angular_size_max_x - angular_size_min_x)
        self.index_of_difficulty = log2(2. * self.angular_distance / self.angular_width)

        self.success = success
        self.first_shot = first_shot
        self.first_hit = first_hit
        self.end_time = end_time

    def to_dict(self) -> Dict:
        return {
            test_name_col: self.test_name,
            movement_name_col: self.movement_name,
            index_of_difficulty_col: self.index_of_difficulty,
            success_col: self.success,
            first_shot_col: self.first_shot,
            first_hit_col: self.first_hit,
            end_time_col: self.end_time
        }


def vis():
    bot_timing_df = pd.read_csv(bot_timing_path)
    bot_event_timing_df = pd.read_csv(bot_event_timing_path)
    human_timing_df = pd.read_csv(human_timing_path)
    human_event_timing_df = pd.read_csv(human_event_timing_path)

    test_names = bot_timing_df.loc[:, test_name_col].unique().tolist()
    test_ids = bot_timing_df.loc[:, test_id_col].unique().tolist()

    bot_data: List[TestData] = []
    human_data: List[TestData] = []

    def get_first_event_val(event_timing_df: pd.DataFrame, test_id: int, event_name: str, value_col: str):
        return event_timing_df[
            (event_timing_df[test_id_col] == test_id) &
            (event_timing_df[event_name_col] == event_name)
        ].loc[:, value_col].values[0]

    for test_name, test_id in zip(test_names, test_ids):
        movement_name: str = "invalid"
        # TODO: fix naming to remove this
        for i, option in enumerate(movement_options_a):
            if option in test_name:
                movement_name = movement_names[i]
        for i, option in enumerate(movement_options_b):
            if option in test_name:
                movement_name = movement_names[i]

        bot_success: bool = \
            bot_timing_df[(bot_timing_df[test_name_col] == test_name)].loc[:, success_col].values[0] > 0
        bot_end_time: float = \
            bot_timing_df[(bot_timing_df[test_name_col] == test_name)].loc[:, end_time_col].values[0]
        bot_first_shot = INVALID_TIME
        bot_first_hit = INVALID_TIME
        if bot_success:
            bot_first_shot = get_first_event_val(bot_event_timing_df, test_id, weapon_fire_event, event_time_col)
            bot_first_hit = get_first_event_val(bot_event_timing_df, test_id, hurt_event, event_time_col)

        bot_data.append(TestData(
            test_name,
            movement_name,
            get_first_event_val(bot_event_timing_df, test_id, angular_distance_x_event, payload_col),
            get_first_event_val(bot_event_timing_df, test_id, angular_distance_y_event, payload_col),
            get_first_event_val(bot_event_timing_df, test_id, angular_size_min_x_event, payload_col),
            get_first_event_val(bot_event_timing_df, test_id, angular_size_min_y_event, payload_col),
            get_first_event_val(bot_event_timing_df, test_id, angular_size_max_x_event, payload_col),
            get_first_event_val(bot_event_timing_df, test_id, angular_size_max_y_event, payload_col),
            bot_success,
            bot_first_shot,
            bot_first_hit,
            bot_end_time
        ))

        human_success: bool = \
            human_timing_df[(human_timing_df[test_name_col] == test_name)].loc[:, success_col].values[0] > 0
        human_end_time: float = \
            human_timing_df[(human_timing_df[test_name_col] == test_name)].loc[:, end_time_col].values[0]
        human_first_shot = INVALID_TIME
        human_first_hit = INVALID_TIME
        if human_success:
            human_first_shot = get_first_event_val(human_event_timing_df, test_id, weapon_fire_event, event_time_col)
            human_first_hit = get_first_event_val(human_event_timing_df, test_id, hurt_event, event_time_col)

        human_data.append(copy.deepcopy(bot_data[-1]))
        human_data[-1].success = human_success
        human_data[-1].first_shot = human_first_shot
        human_data[-1].first_hit = human_first_hit
        human_data[-1].end_time = human_end_time


    bot_data_df = pd.DataFrame([d.to_dict() for d in bot_data])
    human_data_df = pd.DataFrame([d.to_dict() for d in human_data])

    bot_success_data_df = bot_data_df[bot_data_df[success_col]]
    bot_success_tracker_data_df = bot_data_df.loc[:, [test_name_col, success_col]]
    human_success_data_df = human_data_df[human_data_df[success_col]]
    human_success_tracker_subset_data_df = human_data_df.loc[:, [test_name_col, success_col]]
    success_subset_data_df = bot_success_tracker_data_df.merge(human_success_tracker_subset_data_df, on=test_name_col)

    print(f"Bot Success Rate: {len(bot_success_data_df) / len(bot_data_df)}, "
          f"Human Success Rate: {len(human_success_data_df) / len(human_data_df)}")
    success_subset_data_df.to_csv(success_path)

    bot_success_data_df = bot_success_data_df.sort_values(by=index_of_difficulty_col)
    human_success_data_df = human_success_data_df.sort_values(by=index_of_difficulty_col)

    fig = Figure(figsize=(12., 8.), dpi=100)

    bot_shot_ax = fig.add_subplot(2, 3, 1)
    bot_shot_ax.set_title("Bot First Shot")
    bot_shot_ax.set_xlabel(index_of_difficulty_col)
    bot_shot_ax.set_ylabel("Time (s)")

    bot_hit_ax = fig.add_subplot(2, 3, 2)
    bot_hit_ax.set_title("Bot First Hit")
    bot_hit_ax.set_xlabel(index_of_difficulty_col)

    bot_success_ax = fig.add_subplot(2, 3, 3)
    bot_success_ax.set_title("Bot Success")
    bot_success_ax.set_xlabel(index_of_difficulty_col)

    human_shot_ax = fig.add_subplot(2, 3, 4)
    human_shot_ax.set_title("Human First Shot")
    human_shot_ax.set_xlabel(index_of_difficulty_col)
    human_shot_ax.set_ylabel("Time (s)")

    human_hit_ax = fig.add_subplot(2, 3, 5)
    human_hit_ax.set_title("Human First Hit")
    human_hit_ax.set_xlabel(index_of_difficulty_col)

    human_success_ax = fig.add_subplot(2, 3, 6)
    human_success_ax.set_title("Human Success")
    human_success_ax.set_xlabel(index_of_difficulty_col)

    for movement_name in movement_names:
        bot_subset_df = bot_success_data_df[bot_success_data_df[movement_name_col] == movement_name]
        bot_shot_ax.plot(bot_subset_df.loc[:, index_of_difficulty_col],
                         bot_subset_df.loc[:, first_shot_col],
                         linestyle="solid", label=movement_name, marker='o',
                         color=movement_name_to_color[movement_name])
        bot_hit_ax.plot(bot_subset_df.loc[:, index_of_difficulty_col],
                         bot_subset_df.loc[:, first_hit_col],
                         linestyle="solid", label=movement_name, marker='o',
                         color=movement_name_to_color[movement_name])
        bot_success_ax.plot(bot_subset_df.loc[:, index_of_difficulty_col],
                            bot_subset_df.loc[:, end_time_col],
                            linestyle="solid", label=movement_name, marker='o',
                            color=movement_name_to_color[movement_name])

        human_subset_df = human_success_data_df[human_success_data_df[movement_name_col] == movement_name]
        human_shot_ax.plot(human_subset_df.loc[:, index_of_difficulty_col],
                         human_subset_df.loc[:, first_shot_col],
                         linestyle="solid", label=movement_name, marker='o',
                         color=movement_name_to_color[movement_name])
        human_hit_ax.plot(human_subset_df.loc[:, index_of_difficulty_col],
                          human_subset_df.loc[:, first_hit_col],
                          linestyle="solid", label=movement_name, marker='o',
                          color=movement_name_to_color[movement_name])
        human_success_ax.plot(human_subset_df.loc[:, index_of_difficulty_col],
                              human_subset_df.loc[:, end_time_col],
                              linestyle="solid", label=movement_name, marker='o',
                              color=movement_name_to_color[movement_name])
    bot_success_ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.tight_layout()

    fig.savefig(plot_path)


if __name__ == "__main__":
    vis()