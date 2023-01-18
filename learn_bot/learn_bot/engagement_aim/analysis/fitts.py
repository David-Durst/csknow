from typing import Dict

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from math import pow, sqrt, fabs

bot_timing_path = Path(__file__).parent / "test_timing_data" / "bot" / "test_timing.csv"
bot_event_timing_path = Path(__file__).parent / "test_timing_data" / "bot" / "test_event_timing.csv"
human_timing_path = Path(__file__).parent / "test_timing_data" / "human" / "test_timing.csv"
human_event_timing_path = Path(__file__).parent / "test_timing_data" / "human" / "test_event_timing.csv"

# use ramp so can isolate left/right movement coming right after ramp from left/right looking that comes later
moving_options = ["RampNone", "RampForward", "RampLeft", "RampRight"]

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

class TestData:
    angular_distance_x: float
    angular_distance_y: float
    angular_size_min_x: float
    angular_size_min_y: float
    angular_size_max_x: float
    angular_size_max_y: float

    angular_distance: float
    angular_size: float

    success: bool
    first_shot: float
    first_hit: float

    def __init__(self, angular_distance_x: float, angular_distance_y: float,
                 angular_size_min_x: float, angular_size_min_y: float,
                 angular_size_max_x: float, angular_size_max_y: float,
                 success: bool, first_shot: bool, first_hit: bool):
        self.angular_distance_x = angular_distance_x
        self.angular_distance_y = angular_distance_y
        self.angular_size_min_x = angular_size_min_x
        self.angular_size_min_y = angular_size_min_y
        self.angular_size_max_x = angular_size_max_x
        self.angular_size_max_y = angular_size_max_y

        self.angular_distance = sqrt(pow(angular_distance_x, 2.) + pow(angular_distance_y, 2.))
        self.angular_size = \
            fabs(angular_size_max_x - angular_size_min_x) * fabs(angular_size_max_y - angular_size_min_y)

        self.success = success
        self.first_shot = first_shot
        self.first_hit = first_hit


def vis():
    bot_timing_df = pd.read_csv(bot_timing_path)
    bot_event_timing_df = pd.read_csv(bot_event_timing_path)
    human_timing_df = pd.read_csv(human_timing_path)
    human_event_timing_df = pd.read_csv(human_event_timing_path)

    test_names = bot_timing_df.loc[:, test_name_col].unique().tolist()
    test_ids = bot_timing_df.loc[:, test_id_col].unique().tolist()

    # get distance to target in screen space on first frame
    first_event_time_per_test_df = bot_event_timing_df.groupby(test_id_col).agg(min_time=(event_time_col, 'min'))
    first_events_per_test_df = bot_event_timing_df.merge(first_event_time_per_test_df,
                                                         left_on=[test_id_col, event_time_col],
                                                         right_on=[test_id_col, "min_time"], how='inner')

    bot_test_name_to_data: Dict[str, TestData] = {}

    def get_event_val(test_id: int, event_name: str):
        return first_events_per_test_df[
            (first_events_per_test_df[test_id_col] == test_id) &
            (first_events_per_test_df[event_name_col] == event_name)
        ].loc[:, payload_col][0]

    for test_name, test_id in zip(test_names, test_ids):
        success: bool = bot_timing_df[(bot_timing_df[test_name_col] == test_name)].loc[:, success_col][0] > 0
        first_shot = INVALID_TIME
        first_hit = INVALID_TIME
        if success:
            first_shot = 

        bot_test_name_to_data[test_name] = TestData(
            get_event_val(test_id, angular_distance_x_event),
            get_event_val(test_id, angular_distance_y_event),
            get_event_val(test_id, angular_size_min_x_event),
            get_event_val(test_id, angular_size_min_y_event),
            get_event_val(test_id, angular_size_max_x_event),
            get_event_val(test_id, angular_size_max_y_event)
        )

        test_name_to_distance_data[test_name]

    x = 1


    # get size of targets in screen space on first frame


if __name__ == "__main__":
    vis()