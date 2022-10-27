from dataclasses import dataclass
from typing import List


class TemporalIOColumnNames:
    input_columns: List[str]
    output_columns: List[str]
    vis_columns: List[str]

    def __init__(self, base_columns: List[str], prior_ticks: int, cur_tick: int, future_ticks: int):
        self.input_columns = []
        self.output_columns = []
        self.vis_columns = []

        for i in range(prior_ticks, cur_tick+future_ticks):
            offset_str = " (t"
            if i < 0:
                offset_str += str(i)
            elif i > 0:
                offset_str += "+" + str(i)
            offset_str += ")"

            if i < 0:
                for base_col in base_columns:
                    self.input_columns.append(base_col + offset_str)
            else:
                for base_col in base_columns:
                    if i == 0:
                        self.vis_columns.append(base_col + offset_str)
                    self.output_columns.append(base_col + offset_str)

