from dataclasses import dataclass
from typing import List


def get_temporal_field_str(base_str: str, tick: int):
    if tick < 0:
        return f"{base_str} (t-{abs(tick)})"
    elif tick == 0:
        return f"{base_str} (t)"
    else:
        return f"{base_str} (t+{abs(tick)})"


class TemporalIOColumnNames:
    past_columns: List[str]
    present_columns: List[str]
    future_columns: List[str]
    all_columns: List[str]

    def __init__(self, base_columns: List[str], prior_ticks: int, cur_tick: int, future_ticks: int):
        self.past_columns = []
        self.present_columns = []
        self.future_columns = []

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
