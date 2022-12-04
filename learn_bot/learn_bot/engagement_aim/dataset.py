import torch
from torch.utils.data import Dataset
from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, ColumnTypes, PRIOR_TICKS, FUTURE_TICKS, \
    CUR_TICK, DeltaColumn
from typing import List
from learn_bot.libs.temporal_column_names import TemporalIOColumnNames, get_temporal_field_str
from pathlib import Path

data_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'engagementAim.csv'

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
class AimDataset(Dataset):
    def __init__(self, df, cts: IOColumnTransformers):
        self.id = df.loc[:, 'id']
        self.round_id = df.loc[:, 'round id']
        self.tick_id = df.loc[:, 'tick id']
        self.engagement_id = df.loc[:, 'engagement id']
        self.attacker_player_id = df.loc[:, 'attacker player id']
        self.victim_player_id = df.loc[:, 'victim player id']

        round_starts = df.groupby('round id').first('index').loc[:, ['index']].rename(columns={'index': 'start index'})
        round_ends = df.groupby('round id').last('index').loc[:, ['index']].rename(columns={'index': 'end index'})
        self.round_starts_ends = round_starts.join(round_ends, on='round id')

        # convert player id's to indexes
        self.X = torch.tensor(df.loc[:, cts.input_types.column_names()].values).float()
        self.Y = torch.tensor(df.loc[:, cts.output_types.column_names()].values).float()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


engagement_id_column = "engagement id"
tick_id_column = "tick id"
cur_victim_alive_column = "victim alive (t)"
cur_victim_visible_column = "victim visible (t)"
base_hit_victim_column = "hit victim"
base_ticks_since_last_fire_column = "ticks since last fire"
base_ticks_since_last_attack_column = "ticks since last holding attack"

base_abs_x_pos_column = "delta relative first head view angle x"
base_abs_y_pos_column = "delta relative first head view angle y"
base_relative_x_pos_column = "delta relative cur head view angle x"
base_relative_y_pos_column = "delta relative cur head view angle y"
base_recoil_x_column = "scaled recoil angle x"
base_recoil_y_column = "scaled recoil angle y"
base_victim_relative_aabb_min_x = "victim relative cur head min view angle x"
base_victim_relative_aabb_max_x = "victim relative cur head max view angle x"
base_victim_relative_aabb_min_y = "victim relative cur head min view angle y"
base_victim_relative_aabb_max_y = "victim relative cur head max view angle y"

base_float_columns: List[str] = ["attacker view angle x", "attacker view angle y",
                                 "ideal view angle x", "ideal view angle y",
                                 "delta relative first head view angle x", "delta relative first head view angle y",
                                 "delta relative cur head view angle x", "delta relative cur head view angle y",
                                 "hit victim",
                                 "recoil index",
                                 "scaled recoil angle x", "scaled recoil angle y",
                                 "ticks since last fire", "ticks since last holding attack",
                                 "victim visible", "victim alive",
                                 "victim relative first head min view angle x", "victim relative first head min view angle y",
                                 "victim relative first head max view angle x", "victim relative first head max view angle y",
                                 "victim relative first head cur head view angle x", "victim relative first head cur head view angle y",
                                 "victim relative cur head min view angle x", "victim relative cur head min view angle y",
                                 "victim relative cur head max view angle x", "victim relative cur head max view angle y",
                                 "victim relative cur head cur head view angle x", "victim relative cur head cur head view angle y",
                                 "attacker eye pos x", "attacker eye pos y", "attacker eye pos z",
                                 "victim eye pos x", "victim eye pos y", "victim eye pos z",
                                 "attacker vel x", "attacker vel y", "attacker vel z",
                                 "victim vel x", "victim vel y", "victim vel z"]

# some columns only used for output, not input features
base_non_input_float_columns: List[str] = ["ticks until next fire", "ticks until next holding attack"]

non_temporal_float_columns = []

input_categorical_columns: List[str] = ["weapon type"]

temporal_io_float_column_names = TemporalIOColumnNames(base_float_columns, PRIOR_TICKS, CUR_TICK, FUTURE_TICKS)

temporal_o_float_column_names = TemporalIOColumnNames(base_non_input_float_columns, 0, CUR_TICK, FUTURE_TICKS)

input_column_types = ColumnTypes(temporal_io_float_column_names.past_columns + non_temporal_float_columns, [],
                                 input_categorical_columns, [6])

output_relative_x_cols = temporal_io_float_column_names.get_matching_cols(base_abs_x_pos_column, False, True, True)
output_relative_y_cols = temporal_io_float_column_names.get_matching_cols(base_abs_y_pos_column, False, True, True)
output_ref_x_col = get_temporal_field_str(base_abs_x_pos_column, -1)
output_ref_y_col = get_temporal_field_str(base_abs_y_pos_column, -1)
output_delta_x = [DeltaColumn(c, output_ref_x_col) for c in output_relative_x_cols]
output_delta_y = [DeltaColumn(c, output_ref_y_col) for c in output_relative_y_cols]
output_delta = []
for i in range(len(output_delta_x)):
    output_delta.append(output_delta_x[i])
    output_delta.append(output_delta_y[i])
output_standard_cols = temporal_o_float_column_names.get_matching_cols("ticks until", False, True, True)

#output_column_types = ColumnTypes(output_standard_cols, output_delta, [], [])
output_column_types = ColumnTypes([], output_delta, [], [])

seconds_per_tick = 1. / 128. * 1000.
