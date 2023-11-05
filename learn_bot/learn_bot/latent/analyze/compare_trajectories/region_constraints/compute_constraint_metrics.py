from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from PIL import Image, ImageDraw

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import ComparisonConfig
from learn_bot.latent.analyze.test_traces.run_trace_visualization import convert_to_canvas_coordinates
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, PlayerPlaceAreaColumns
from learn_bot.latent.vis.draw_inference import d2_bottom_right_y, d2_top_left_y, d2_bottom_right_x, d2_top_left_x


class PositionConstraintType(Enum):
    XOnly = 1
    YOnly = 2
    XY = 3

@dataclass
class PositionConstraint:
    position_constraint_type: PositionConstraintType
    x_constraint: float
    x_greater_than: bool
    y_constraint: float
    y_greater_than: bool


attack_a_spawn_t_long_constraint = PositionConstraint(PositionConstraintType.YOnly, 0., False, 2350., True)
attack_b_hole_teammate_doors = PositionConstraint(PositionConstraintType.YOnly, 0., False, 2500., True)
defend_a_ct_cat = PositionConstraint(PositionConstraintType.XOnly, 990., False, 0., False)
defend_a_ct_long = PositionConstraint(PositionConstraintType.XOnly, 990., False, 0., False)
defend_b_ct_site = PositionConstraint(PositionConstraintType.YOnly, 0., False, 1480., False)
defend_b_ct_hole = PositionConstraint(PositionConstraintType.XY, -1780., False, 2310., True)


test_name_to_constraint: Dict[str, PositionConstraint] = {
    "AttackASpawnTLong": attack_a_spawn_t_long_constraint,
    "AttackASpawnTLongTwoTeammates": attack_a_spawn_t_long_constraint,
    "AttackASpawnTExtendedA": attack_a_spawn_t_long_constraint,
    "AttackBDoorsTeammateHole": attack_b_hole_teammate_doors,
    "AttackBHoleTeammateBDoors": attack_b_hole_teammate_doors,
    "DefendACTCat": defend_a_ct_cat,
    "DefendACTCatTwoTeammates": defend_a_ct_cat,
    "DefendACTLong": defend_a_ct_long,
    "DefendACTLongWithTeammate": defend_a_ct_long,
    "DefendACTLongWithTwoTeammates": defend_a_ct_long,
    "DefendBCTSite": defend_b_ct_site,
    "DefendBCTTuns": defend_b_ct_site,
    "DefendBCTHole": defend_b_ct_hole,
    "DefendBCTHoleTwoTeammates": defend_b_ct_hole,
}

test_name_column = 'test name'
config_column = 'config'
num_trials_column = 'num trials'
mean_time_constraint_valid_column = 'mean time constraint valid'
std_time_constraint_valid_column = 'std time constraint valid'


@dataclass
class ConstraintResult:
    test_name: str
    num_trials: int
    mean_time_constraint_valid: float
    std_time_constraint_valid: float

    def save(self, save_path: Path, config: str):
        with open(save_path, 'w') as f:
            f.write('test name,config,num trials,mean time constraint valid,std time constraint valid\n')
            f.write(f'{self.test_name},{config},{self.num_trials},'
                    f'{self.mean_time_constraint_valid},{self.std_time_constraint_valid}\n')


def get_player_that_moves_most(round_trajectory_df: pd.DataFrame) -> int:
    player_that_moves_most: int = -1
    farthest_distance: float = -1

    for i, player_place_area_columns in enumerate(specific_player_place_area_columns):
        alive_trajectory_df = round_trajectory_df[round_trajectory_df[player_place_area_columns.alive] == 1]
        if len(alive_trajectory_df) == 0:
            continue
        start_row = round_trajectory_df.iloc[0]
        end_row = round_trajectory_df.iloc[-1]
        player_distance = (
            (end_row[player_place_area_columns.pos[0]] - start_row[player_place_area_columns.pos[0]]) ** 2. +
            (end_row[player_place_area_columns.pos[1]] - start_row[player_place_area_columns.pos[1]]) ** 2. +
            (end_row[player_place_area_columns.pos[2]] - start_row[player_place_area_columns.pos[2]]) ** 2.
        ) ** .5
        if player_distance > farthest_distance:
            farthest_distance = player_distance
            player_that_moves_most = i

    return player_that_moves_most


constraint_line_color = (255, 0, 0, 100)


def plot_constraint_on_img(trajectories_img: Image.Image, constraint: PositionConstraint):
    trajectories_overlay_im = Image.new("RGBA", trajectories_img.size, (255, 255, 255, 0))
    trajectories_img_draw = ImageDraw.Draw(trajectories_overlay_im)
    if constraint.position_constraint_type == PositionConstraintType.XOnly:
        x_constraint_coords = \
            convert_to_canvas_coordinates(pd.Series([constraint.x_constraint, constraint.x_constraint]),
                                          pd.Series([d2_bottom_right_y, d2_top_left_y]))
        trajectories_img_draw.rectangle((x_constraint_coords[0][0] - 5, x_constraint_coords[1][0],
                                         x_constraint_coords[0][1] + 5, x_constraint_coords[1][1]),
                                        fill=constraint_line_color)
    elif constraint.position_constraint_type == PositionConstraintType.YOnly:
        y_constraint_coords = \
            convert_to_canvas_coordinates(pd.Series([d2_bottom_right_x, d2_top_left_x]),
                                          pd.Series([constraint.y_constraint, constraint.y_constraint]))
        trajectories_img_draw.rectangle((y_constraint_coords[0][0], y_constraint_coords[1][0] - 5,
                                         y_constraint_coords[0][1], y_constraint_coords[1][1] + 5),
                                        fill=constraint_line_color)
    else:
        if constraint.x_greater_than:
            min_x = constraint.x_constraint
            max_x = d2_bottom_right_x
        else:
            min_x = d2_top_left_x
            max_x = constraint.x_constraint
        if constraint.y_greater_than:
            min_y = constraint.y_constraint
            max_y = d2_top_left_y
        else:
            min_y = d2_bottom_right_y
            max_y = constraint.y_constraint
        x_constraint_coords = \
            convert_to_canvas_coordinates(pd.Series([min_x, max_x]),
                                          pd.Series([constraint.y_constraint, constraint.y_constraint]))
        trajectories_img_draw.rectangle((x_constraint_coords[0][0], x_constraint_coords[1][0] - 5,
                                         x_constraint_coords[0][1], x_constraint_coords[1][1] + 5),
                                        fill=constraint_line_color)
        y_constraint_coords = \
            convert_to_canvas_coordinates(pd.Series([constraint.x_constraint, constraint.x_constraint]),
                                          pd.Series([min_y, max_y]))
        trajectories_img_draw.rectangle((y_constraint_coords[0][0] - 5, y_constraint_coords[1][0],
                                         y_constraint_coords[0][1] + 5, y_constraint_coords[1][1]),
                                        fill=constraint_line_color)
    trajectories_img.alpha_composite(trajectories_overlay_im)


def check_constraint_metrics(round_trajectory_dfs: List[pd.DataFrame], test_name: str,
                             target_full_table_ids: Optional[List[int]],
                             trajectories_img: Image.Image) -> Optional[ConstraintResult]:
    if test_name not in test_name_to_constraint:
        return None
    constraint = test_name_to_constraint[test_name]
    num_trials = len(round_trajectory_dfs)
    percent_constraint_valid_per_round: List[float] = []

    if target_full_table_ids is None:
        target_full_table_ids = [get_player_that_moves_most(round_trajectory_df)
                                 for round_trajectory_df in round_trajectory_dfs]

    for round_trajectory_df, target_full_table_id in zip(round_trajectory_dfs, target_full_table_ids):
        player_place_area_columns = specific_player_place_area_columns[target_full_table_id]
        if constraint.position_constraint_type == PositionConstraintType.XOnly:
            if constraint.x_greater_than:
                percent_constraint_valid_per_round.append(
                    (round_trajectory_df[player_place_area_columns.pos[0]] > constraint.x_constraint).sum())
            else:
                percent_constraint_valid_per_round.append(
                    (round_trajectory_df[player_place_area_columns.pos[0]] < constraint.x_constraint).sum())
        elif constraint.position_constraint_type == PositionConstraintType.YOnly:
            if constraint.y_greater_than:
                percent_constraint_valid_per_round.append(
                    (round_trajectory_df[player_place_area_columns.pos[1]] > constraint.y_constraint).sum())
            else:
                percent_constraint_valid_per_round.append(
                    (round_trajectory_df[player_place_area_columns.pos[1]] < constraint.y_constraint).sum())
        else:
            if constraint.x_greater_than:
                x_constraint_valid = \
                    (round_trajectory_df[player_place_area_columns.pos[0]] > constraint.x_constraint)
            else:
                x_constraint_valid = \
                    (round_trajectory_df[player_place_area_columns.pos[0]] < constraint.x_constraint)
            if constraint.y_greater_than:
                constraint_valid = x_constraint_valid & \
                                   (round_trajectory_df[player_place_area_columns.pos[1]] > constraint.y_constraint)
            else:
                constraint_valid = x_constraint_valid & \
                                   (round_trajectory_df[player_place_area_columns.pos[1]] < constraint.y_constraint)
            percent_constraint_valid_per_round.append(constraint_valid.sum())

    percent_constraint_valid_per_round_series: pd.Series = pd.Series(percent_constraint_valid_per_round)
    plot_constraint_on_img(trajectories_img, constraint)

    return ConstraintResult(test_name, num_trials, percent_constraint_valid_per_round_series.mean(),
                            percent_constraint_valid_per_round_series.std())
