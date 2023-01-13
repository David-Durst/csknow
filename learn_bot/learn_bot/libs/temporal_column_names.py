from dataclasses import dataclass
from typing import List


def get_temporal_field_str(base_str: str, tick: int):
    if tick < 0:
        return f"{base_str} (t-{abs(tick)})"
    elif tick == 0:
        return f"{base_str} (t)"
    else:
        return f"{base_str} (t+{abs(tick)})"


def get_base_field_str(temporal_str: str):
    if " (t" in temporal_str:
       end_index = temporal_str.index(" (t")
       return temporal_str[0:end_index]
    else:
        return temporal_str


class TemporalIOColumnNames:
    past_columns: List[str]
    present_columns: List[str]
    future_columns: List[str]
    all_columns: List[str]
    prior_ticks: int
    cur_tick: int
    future_ticks: int

    def __init__(self, base_columns: List[str], prior_ticks: int, cur_tick: int, future_ticks: int):
        self.past_columns = []
        self.present_columns = []
        self.future_columns = []
        self.prior_ticks = prior_ticks
        self.cur_tick = cur_tick
        self.future_ticks = future_ticks

        for i in range(prior_ticks, cur_tick+future_ticks):
            if i < 0:
                for base_col in base_columns:
                    self.past_columns.append(get_temporal_field_str(base_col, i))
            elif i == 0:
                for base_col in base_columns:
                    self.present_columns.append(get_temporal_field_str(base_col, i))
            else:
                for base_col in base_columns:
                    self.future_columns.append(get_temporal_field_str(base_col, i))

        self.all_columns = self.past_columns + self.present_columns + self.future_columns

    def get_matching_cols(self, match_str, include_past=True, include_present=True, include_future=True):
        results = []
        if include_past:
            results += [c for c in self.past_columns if match_str in c]
        if include_present:
            results += [c for c in self.present_columns if match_str in c]
        if include_future:
            results += [c for c in self.future_columns if match_str in c]
        return results

    def get_num_cats_per_temporal_column(self, num_cats_per_base_column: List[int],
                                         include_past=True, include_present=True, include_future=True) -> List[int]:
        result = []
        if include_past:
            for i in range(self.prior_ticks, 0):
                result += num_cats_per_base_column
        if include_present:
            for i in range(0, self.cur_tick):
                result += num_cats_per_base_column
        if include_future:
            for i in range(self.cur_tick, self.cur_tick+self.future_ticks):
                result += num_cats_per_base_column
        return result
