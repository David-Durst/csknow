from dataclasses import dataclass
from enum import Enum, auto
from math import log, ceil
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_and_events import title_font
from learn_bot.latent.analyze.plot_trajectory_heatmap.compute_teamwork_metrics import compute_teamwork_metrics, \
    clear_teamwork_title_caches
from learn_bot.libs.pil_helpers import concat_horizontal, concat_vertical
from learn_bot.latent.analyze.knn.plot_min_distance_rounds import game_tick_rate
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.plot_trajectory_heatmap.split_discontinuities import \
    identify_discontinuity_indices_in_valid_points, split_list_by_valid_points
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, convert_to_canvas_coordinates, \
    bot_ct_color_list, replay_ct_color_list, bot_t_color_list, replay_t_color_list
from learn_bot.latent.engagement.column_names import round_id_column, game_tick_number_column
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, PlayerPlaceAreaColumns
from learn_bot.latent.vis.draw_inference import d2_bottom_right_x, d2_bottom_right_y, d2_top_left_x, d2_top_left_y
from learn_bot.libs.io_transforms import CPU_DEVICE_STR
from learn_bot.libs.vec import Vec3


class ImageBuffers:
    ct_buffer: np.ndarray
    t_buffer: np.ndarray

    def __init__(self):
        self.ct_buffer = np.zeros(d2_img.size, dtype=np.intc)
        self.t_buffer = np.zeros(d2_img.size, dtype=np.intc)

    def get_buffer(self, ct_team) -> np.ndarray:
        if ct_team:
            return self.ct_buffer
        else:
            return self.t_buffer

# engagement stats columns
time_to_kill_col = 'Time To Kill'
time_to_shoot_col = 'Time To Shoot'
crosshair_distance_to_enemy_col = 'Crosshair Distance To Enemy'

spread_radius = 2
title_to_line_buffers: Dict[str, ImageBuffers] = {}
title_to_point_buffers: Dict[str, ImageBuffers] = {}
title_to_num_trajectory_ids: Dict[str, int] = {}
title_to_num_points: Dict[str, int] = {}
title_to_lifetimes: Dict[str, List[float]] = {}
title_to_speeds: Dict[str, List[float]] = {}
title_to_shots_per_kill: Dict[str, List[float]] = {}
title_to_engage: Dict[str, List[Dict[str, float]]] = {}
title_to_key_events: Dict[str, int] = {}
title_to_team_to_pos_dict = Dict[str, Dict[bool, Tuple[List[float], List[float]]]]
# this is different from buffers as buffers stores each point exactly once
# this can store a point multiple times if multiple events happened on that point (need to weight it)
# such as shooting multiple bullets in same point (as span time over multiple shots in 64ms)
title_to_team_to_key_event_pos: title_to_team_to_pos_dict = {}


def get_title_to_line_buffers() -> Dict[str, ImageBuffers]:
    return title_to_line_buffers


def get_title_to_point_buffers() -> Dict[str, ImageBuffers]:
    return title_to_point_buffers


def get_title_to_num_trajectory_ids() -> Dict[str, int]:
    return title_to_num_trajectory_ids


def get_title_to_num_points() -> Dict[str, int]:
    return title_to_num_points


def get_title_to_lifetimes() -> Dict[str, List[float]]:
    return title_to_lifetimes


def get_title_to_speeds() -> Dict[str, List[float]]:
    return title_to_speeds


def get_title_to_shots_per_kill() -> Dict[str, List[float]]:
    return title_to_shots_per_kill


def get_title_to_key_events() -> Dict[str, int]:
    return title_to_key_events


def get_title_to_team_to_key_event_pos() -> title_to_team_to_pos_dict:
    return title_to_team_to_key_event_pos


def clear_title_caches():
    global title_to_line_buffers, title_to_point_buffers, title_to_num_trajectory_ids, title_to_num_points, \
        title_to_lifetimes, title_to_speeds, \
        title_to_shots_per_kill, title_to_key_events, title_to_team_to_key_event_pos
    title_to_line_buffers = {}
    title_to_point_buffers = {}
    title_to_num_trajectory_ids = {}
    title_to_num_points = {}
    title_to_lifetimes = {}
    title_to_speeds = {}
    title_to_shots_per_kill = {}
    title_to_key_events = {}
    title_to_team_to_key_event_pos = {}
    clear_teamwork_title_caches()


# data validation function, making sure I didn't miss any kill/killed/shots events
def compute_overall_key_event_indices(vis_df: pd.DataFrame,
                                      trajectory_filter_options: TrajectoryFilterOptions) -> Tuple[Set[int], int]:
    key_event_indices = set()
    num_key_events = 0
    for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
        event_constraint = None
        if trajectory_filter_options.only_kill:
            event_constraint = vis_df[player_place_area_columns.player_kill_next_tick] > 0.5
            num_key_events += int(vis_df[player_place_area_columns.player_kill_next_tick].sum())
        elif trajectory_filter_options.only_killed:
            event_constraint = vis_df[player_place_area_columns.player_killed_next_tick] > 0.5
            num_key_events += int(vis_df[player_place_area_columns.player_killed_next_tick].sum())
        elif trajectory_filter_options.only_shots:
            event_constraint = vis_df[player_place_area_columns.player_shots_cur_tick] > 0.5
            num_key_events += int(vis_df[player_place_area_columns.player_shots_cur_tick].sum())
        if event_constraint is not None:
            key_event_indices.update(vis_df[event_constraint].index.tolist())
    return key_event_indices, num_key_events


# debugging to make sure that get all events even with filtering
# this requires disabling restricting to push only round ids, as want to tie out with overall count numbers
debug_event_counting = False


def get_debug_event_counting() -> bool:
    return debug_event_counting


def plot_gaussian(buffer: np.ndarray, row: int, col: int, num_points: int):
    for r in range(row-1, row+2):
        for c in range(col-1, col+2):
            weight = num_points
            if r == row:
                weight *= 2
            if c == col:
                weight *= 2
            buffer[min(max(r, 0), buffer.shape[0]-1), min(max(c, 0), buffer.shape[1]-1)] += weight


def plot_one_trajectory_dataset(loaded_model: LoadedModel, id_df: pd.DataFrame, vis_df: pd.DataFrame,
                                dataset: np.ndarray, trajectory_filter_options: TrajectoryFilterOptions, title: str):
    if title not in title_to_line_buffers:
        title_to_line_buffers[title] = ImageBuffers()
        title_to_point_buffers[title] = ImageBuffers()
        title_to_num_points[title] = 0

    if trajectory_filter_options.trajectory_counter is None:
        trajectory_ids = id_df[round_id_column].unique()
        if trajectory_filter_options.valid_round_ids is not None:
            if debug_event_counting:
                trajectory_ids = id_df[round_id_column].unique().tolist()
            else:
                trajectory_ids = [r for r in trajectory_ids if r in trajectory_filter_options.valid_round_ids]
        trajectory_id_col = id_df[round_id_column]
    else:
        trajectory_ids = trajectory_filter_options.trajectory_counter.unique()
        trajectory_id_col = trajectory_filter_options.trajectory_counter
        # trajectories are already short (open sim runs for 5s), no need to restrict them further
        assert trajectory_filter_options.round_game_seconds is None

    if title not in title_to_num_trajectory_ids:
        title_to_num_trajectory_ids[title] = 0
    title_to_num_trajectory_ids[title] += len(trajectory_ids)

    weapon_scoped_to_max_speed = loaded_model.model.weapon_scoped_to_max_speed_tensor_gpu.to(CPU_DEVICE_STR)

    if debug_event_counting:
        # compute key events before any filtering/spltting by trajectory to ensure logic is correct
        # overall - look at entire df, per trajectory - compute by looking at each trajectory per player after alive filter
        overall_key_event_indices, overall_num_key_events = \
            compute_overall_key_event_indices(vis_df, trajectory_filter_options)
        if title not in title_to_key_events:
            title_to_key_events[title] = 0
        title_to_key_events[title] += overall_num_key_events
        per_trajectory_key_event_indices = set()

    for trajectory_id in trajectory_ids:
        trajectory_np = dataset[trajectory_id_col == trajectory_id]
        trajectory_id_df = id_df[trajectory_id_col == trajectory_id]
        trajectory_vis_df = vis_df[trajectory_id_col == trajectory_id]
        first_game_tick_number = trajectory_id_df[game_tick_number_column].iloc[0]

        # if no filter, compute the teamwork metrics
        if trajectory_filter_options.is_no_filter():
            compute_teamwork_metrics(loaded_model, trajectory_np, trajectory_vis_df, title)

        # early terminate if requiring all players to be in regions and one fails
        num_out_bounds_ct = 0
        num_out_bounds_t = 0
        num_in_bounds_ct = 0
        num_in_bounds_t = 0
        if trajectory_filter_options.team_based_all_start_in_region is not None:
            # make sure bomb planted in in right place for filter
            planted_a = trajectory_np[0, loaded_model.model.c4_planted_columns[0]]
            planted_b = trajectory_np[0, loaded_model.model.c4_planted_columns[1]]
            assert planted_a or planted_b
            if (planted_a and not trajectory_filter_options.team_based_all_start_in_region.bomb_planted_a) or \
                    (planted_b and trajectory_filter_options.team_based_all_start_in_region.bomb_planted_a):
                continue

            # default is true as will and all of below conditions
            all_players_in_region = trajectory_np[:, 0] == trajectory_np[:, 0]

            for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
                # dead players can't fail a starting condition test
                ct_team = team_strs[0] in player_place_area_columns.player_id
                player_pos_x_np = trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 0]]
                player_pos_y_np = trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 1]]
                player_pos_z_np = trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 2]]
                constraint_aabb = trajectory_filter_options.team_based_all_start_in_region.ct_region if ct_team \
                    else trajectory_filter_options.team_based_all_start_in_region.t_region
                all_players_in_region = all_players_in_region & (
                        (trajectory_np[:, loaded_model.model.alive_columns[player_index]] < 0.5) |
                        ((player_pos_x_np >= constraint_aabb.min.x) &
                         (player_pos_x_np <= constraint_aabb.max.x) &
                         (player_pos_y_np >= constraint_aabb.min.y) &
                         (player_pos_y_np <= constraint_aabb.max.y) &
                         (player_pos_z_np >= constraint_aabb.min.z) &
                         (player_pos_z_np <= constraint_aabb.max.z)))

            # let any moment be the start of when all players are in region
            all_players_been_in_region = np.cumsum(all_players_in_region) > 0
            trajectory_np = trajectory_np[all_players_been_in_region]
            #trajectory_np = trajectory_np[all_players_in_region]

        # restrict to start of round if filtering based on time
        if trajectory_filter_options.round_game_seconds is not None:
            start_condition = (trajectory_id_df[game_tick_number_column] - first_game_tick_number) >= \
                              game_tick_rate * trajectory_filter_options.round_game_seconds.start
            stop_condition = (trajectory_id_df[game_tick_number_column] - first_game_tick_number) <= \
                              game_tick_rate * trajectory_filter_options.round_game_seconds.stop
            trajectory_np = trajectory_np[start_condition & stop_condition]

        # require one player in region if allowing all players when one in region
        if trajectory_filter_options.include_all_players_when_one_in_region:
            player_in_region = trajectory_np[:, 0] != trajectory_np[:, 0]
            for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
                # require offense only, as they set the positioning for everyone else
                ct_team = team_strs[0] in player_place_area_columns.player_id
                if not ct_team:
                    continue
                player_pos_x_np = trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 0]]
                player_pos_y_np = trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 1]]
                player_in_region = player_in_region | (
                    (trajectory_np[:, loaded_model.model.alive_columns[player_index]] > 0.5) &
                    (player_pos_x_np >= trajectory_filter_options.player_starts_in_region.min.x) &
                    (player_pos_x_np <= trajectory_filter_options.player_starts_in_region.max.x) &
                    (player_pos_y_np >= trajectory_filter_options.player_starts_in_region.min.y) &
                    (player_pos_y_np <= trajectory_filter_options.player_starts_in_region.max.y))
            # within a trajectory, players can leave and enter the region of interest, creating discontinuous trajectories
            # identify those splits so can plot them separately
            # can't happen with time because only have one continuous time region per plot
            trajectory_valid_discontinuities = identify_discontinuity_indices_in_valid_points(player_in_region)
        else:
            player_in_region = trajectory_np[:, 0] == trajectory_np[:, 0]
            trajectory_valid_discontinuities = None
        trajectory_np = trajectory_np[player_in_region]

        if trajectory_filter_options.filtering_key_events() and \
                (trajectory_filter_options.round_game_seconds is not None or
                 trajectory_filter_options.include_all_players_when_one_in_region):
            raise Exception("can't filter by game seconds or player in region and only key events like kill/killed/shot")
        if trajectory_filter_options.computing_metrics() and \
                (trajectory_filter_options.round_game_seconds is not None or
                 trajectory_filter_options.include_all_players_when_one_in_region):
            raise Exception("can't filter by game seconds or player in region and compute metrics")

        for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
            ct_team = team_strs[0] in player_place_area_columns.player_id

            alive_constraint = trajectory_np[:, loaded_model.model.alive_columns[player_index]] == 1

            # don't worry about dead players
            if sum(alive_constraint) == 0:
                continue

            alive_trajectory_np = trajectory_np[alive_constraint]
            # only using vis_df to compute speed and filter for key events, so don't filter it if not computing metrics
            # or filtering for key events, otherwise will have length mismatch as other types of filters (like area
            # and time) will make trajectory_np have different length
            if trajectory_filter_options.computing_metrics() or trajectory_filter_options.filtering_key_events():
                alive_trajectory_vis_df = trajectory_vis_df[alive_constraint]
            else:
                alive_trajectory_vis_df = None

            # require start location if filtering based on starting region to only one player
            first_pos = (
                alive_trajectory_np[0, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 0]],
                alive_trajectory_np[0, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 1]]
            )
            if trajectory_filter_options.player_starts_in_region is not None and \
                    not trajectory_filter_options.include_all_players_when_one_in_region and \
                    not trajectory_filter_options.start_constraint_applies_to_all and (
                    first_pos[0] < trajectory_filter_options.player_starts_in_region.min.x or
                    first_pos[0] > trajectory_filter_options.player_starts_in_region.max.x or
                    first_pos[1] < trajectory_filter_options.player_starts_in_region.min.y or
                    first_pos[1] > trajectory_filter_options.player_starts_in_region.max.y):
                continue

            if trajectory_filter_options.only_kill + trajectory_filter_options.only_killed + \
                    trajectory_filter_options.only_shots > 1:
                raise Exception("can only filter for one type of key event at a time")
            if trajectory_filter_options.only_kill:
                event_series = alive_trajectory_vis_df[player_place_area_columns.player_kill_next_tick]
            elif trajectory_filter_options.only_killed:
                event_series = alive_trajectory_vis_df[player_place_area_columns.player_killed_next_tick]
            elif trajectory_filter_options.only_killed_or_end:
                # either killed next tick, or alive on the last tick
                # note: a few times player dies without kill event. this is a core flaw in demo file recording
                # not going to invent kills, but want to catch ends here, so look for end of trajectory for this player
                # rather than end of overall round
                event_series = np.maximum(alive_trajectory_vis_df[player_place_area_columns.player_killed_next_tick],
                                          alive_trajectory_vis_df.index == alive_trajectory_vis_df.index[-1])
                # 12_29_23_model_learned_no_time_with_partial_1474_rounds_0
                # has a late round end event, resulting in an extra tick with everyone alive after dead
                # this is a bad demo, not my fault, can't do anything about bad round end events,
                # print for that one case
                # 1_8_24_learned_similarity_save has it too
                #if '12_29_23_model_learned_no_time_with_partial_1474_rounds_0' in \
                #    str(loaded_model.dataset.data_hdf5s[loaded_model.cur_hdf5_index].hdf5_path):
                #    if sum(event_series) != 1.:
                #        print('extra tick')
                #else:
                #    # in rare situations,
                sum_event_series = sum(event_series)
                if sum_event_series == 2.:
                    print("bad demo with late round end event")
                assert sum(event_series) == 1. or sum(event_series) == 2.
            elif trajectory_filter_options.only_shots:
                event_series = alive_trajectory_vis_df[player_place_area_columns.player_shots_cur_tick]
            else:
                event_series = None
            # multiple shots can occur in the same tick,
            # so need to recover number of events per tick when event happened
            num_events_per_tick_with_event = None
            if event_series is not None:
                event_constraint = event_series > 0.5
                alive_trajectory_np = alive_trajectory_np[event_constraint]
                alive_trajectory_vis_df = alive_trajectory_vis_df[event_constraint]
                num_events_per_tick_with_event = event_series[event_constraint].tolist()
                if debug_event_counting:
                    per_trajectory_key_event_indices.update(alive_trajectory_vis_df.index.tolist())
            if trajectory_filter_options.compute_lifetimes and \
                    (trajectory_filter_options.only_kill or trajectory_filter_options.only_killed or
                     trajectory_filter_options.only_shots):
                raise Exception("can't filter to kill/killed/shot events and compute lifetimes")

            x_pos = alive_trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 0]]
            y_pos = alive_trajectory_np[:, loaded_model.model.nested_players_pos_columns_tensor[player_index, 0, 1]]
            canvas_pos_np = convert_to_canvas_coordinates(x_pos, y_pos)
            canvas_pos_x_np = canvas_pos_np[0].astype(np.intc)
            canvas_pos_y_np = canvas_pos_np[1].astype(np.intc)
            canvas_pos_xy = list(zip(list(canvas_pos_x_np), list(canvas_pos_y_np)))

            point_buffer = title_to_point_buffers[title].get_buffer(ct_team)

            line_buffer = title_to_line_buffers[title].get_buffer(ct_team)
            if trajectory_filter_options.filtering_key_events():
                for i, pos_xy in enumerate(canvas_pos_xy):
                    # drawing not useful as so few points and no lines connecting them
                    #plot_gaussian(point_buffer, pos_xy[1], pos_xy[0], num_events_per_tick_with_event[i])
                    point_buffer[pos_xy[1], pos_xy[0]] += num_events_per_tick_with_event[i]
                    if title not in title_to_team_to_key_event_pos:
                        title_to_team_to_key_event_pos[title] = {}
                    if ct_team not in title_to_team_to_key_event_pos[title]:
                        title_to_team_to_key_event_pos[title][ct_team] = ([], [])
                    # add multiple entries for ticks with multiple copies of each event
                    for _ in range(int(num_events_per_tick_with_event[i])):
                        title_to_team_to_key_event_pos[title][ct_team][0].append(x_pos[i])
                        title_to_team_to_key_event_pos[title][ct_team][1].append(y_pos[i])
            else:
                # sometimes get a late round end signal and so everyone dies and teleports, ignore those last few ticks
                for i in range(len(canvas_pos_xy) - 1):
                    if ((canvas_pos_xy[i][0] - canvas_pos_xy[i + 1][0]) ** 2. +
                        (canvas_pos_xy[i][1] - canvas_pos_xy[i + 1][1]) ** 2.) ** 0.5 > 100:
                        assert i > 10 and len(canvas_pos_xy) - i < 5
                        canvas_pos_xy = canvas_pos_xy[:i+1]
                for pos_xy in canvas_pos_xy:
                    #plot_gaussian(point_buffer, pos_xy[1], pos_xy[0], 1)
                    point_buffer[pos_xy[1], pos_xy[0]] += 1
                cur_player_d2_img = Image.new("L", d2_img.size, color=0)
                # blending occurs across images, so want to keep same number of images (1 per player trajectory)
                # no matter how many discontinuities occur in each player's trajectory
                # can differ per player even in same round due to early deaths
                cur_player_d2_drw = ImageDraw.Draw(cur_player_d2_img)
                for continuous_canvas_pos_xy in split_list_by_valid_points(canvas_pos_xy,
                                                                           trajectory_valid_discontinuities):
                    cur_player_d2_drw.line(xy=continuous_canvas_pos_xy, fill=1, width=5)
                line_buffer += np.asarray(cur_player_d2_img)
            title_to_num_points[title] += len(alive_trajectory_np)
            compute_lifetimes(trajectory_id_df, alive_trajectory_np, first_game_tick_number, trajectory_filter_options,
                              title)
            compute_speeds(alive_trajectory_vis_df, player_place_area_columns, weapon_scoped_to_max_speed,
                           trajectory_filter_options, title)
            compute_shots_per_kill(alive_trajectory_vis_df, player_place_area_columns, trajectory_filter_options, title)



    # verify that got all key events, no need to check killed or end as have assert above for that
    # which ensures every trajectory has 1
    if debug_event_counting and trajectory_filter_options.filtering_key_events() and \
            not trajectory_filter_options.only_killed_or_end:
        per_trajectory_indices_not_in_overall = per_trajectory_key_event_indices.difference(overall_key_event_indices)
        if not len(per_trajectory_indices_not_in_overall) == 0:
            print(f"How is per trajectory getting extra indices {per_trajectory_indices_not_in_overall}")
        overall_indices_not_in_per_trajectory = overall_key_event_indices.difference(per_trajectory_key_event_indices)
        if not len(overall_indices_not_in_per_trajectory) == 0:
            print("overall has more indices")
            print(f"hdf5 index {loaded_model.cur_hdf5_index}")
            print("missing entries in id_df")
            print(id_df.loc[list(overall_indices_not_in_per_trajectory)])
            print("first complete missing entry")
            print(id_df.loc[list(overall_indices_not_in_per_trajectory)].iloc[0])
        per_trajectory_num_key_events = len(title_to_team_to_key_event_pos[title][True][0]) + \
                                        len(title_to_team_to_key_event_pos[title][False][0])
        if title_to_key_events[title] != per_trajectory_num_key_events:
            print("overall and per trajectory different number of events")
            print(f"hdf5 index {loaded_model.cur_hdf5_index}")
            print(f"overall num key events {title_to_key_events[title]}")
            print(f"per trajectory num key events {per_trajectory_num_key_events}")


def compute_lifetimes(trajectory_id_df: pd.DataFrame, alive_trajectory_np: np.ndarray, first_game_tick_number: int,
                      trajectory_filter_options: TrajectoryFilterOptions, title: str):
    if trajectory_filter_options.compute_lifetimes:
        if title not in title_to_lifetimes:
            title_to_lifetimes[title] = []
        lifetime_in_game_ticks = trajectory_id_df[game_tick_number_column].iloc[len(alive_trajectory_np) - 1] - \
                                 first_game_tick_number
        # sometimes if 41 seconds due to alignment issues, just make it 40 in those cases to handle
        # weird counting issues
        title_to_lifetimes[title].append(min(lifetime_in_game_ticks / game_tick_rate, 40.))


def compute_speeds(alive_trajectory_vis_df: pd.DataFrame, player_place_area_columns: PlayerPlaceAreaColumns,
                   weapon_scoped_to_max_speed: torch.Tensor, trajectory_filter_options: TrajectoryFilterOptions,
                   title: str):
    if trajectory_filter_options.compute_speeds:
        if title not in title_to_speeds:
            title_to_speeds[title] = []
        # z speed is out of our control, only use x and y
        # z isn't tracked when running up a hill, its about jumping speed, which isn't part of wasd speed
        speeds = (alive_trajectory_vis_df[player_place_area_columns.vel[0:2]] ** 2.).sum(axis=1) ** 0.5
        weapon_id_index = alive_trajectory_vis_df[player_place_area_columns.player_weapon_id]
        scoped_index = alive_trajectory_vis_df[player_place_area_columns.player_scoped]
        weapon_id_and_scoped_index = torch.tensor((weapon_id_index * 2 + scoped_index).astype('int').values)
        # mul weapon index by 2 as inner dimension is scoped, and 2 options for scoping (scoped or unscoped)
        max_speed_per_weapon_and_scoped = \
            torch.index_select(weapon_scoped_to_max_speed, 0, weapon_id_and_scoped_index)
        scaled_speeds = speeds.values / max_speed_per_weapon_and_scoped.numpy()
        # found 1.3% are over max speed, just clip them to avoid annoyances
        clipped_scaled_speeds = np.clip(scaled_speeds, 0., 1.)
        title_to_speeds[title] += clipped_scaled_speeds.tolist()


def compute_shots_per_kill(alive_trajectory_vis_df: pd.DataFrame, player_place_area_columns: PlayerPlaceAreaColumns,
                           trajectory_filter_options: TrajectoryFilterOptions, title: str):
    if trajectory_filter_options.compute_shots_per_kill:
        if title not in title_to_shots_per_kill:
            title_to_shots_per_kill[title] = []
        num_shots = alive_trajectory_vis_df[player_place_area_columns.player_shots_cur_tick].sum()
        num_kills = alive_trajectory_vis_df[player_place_area_columns.player_kill_next_tick].sum()
        # metric undefined if player fails to score a kill
        if num_kills > 0:
            title_to_shots_per_kill[title].append(num_shots / num_kills)


def compute_crosshair_distance_to_engage(trajectory_id_df: pd.DataFrame, alive_trajectory_vis_df: pd.DataFrame,
                                         player_place_area_columns: PlayerPlaceAreaColumns,
                                         trajectory_filter_options: TrajectoryFilterOptions, title: str):
    if trajectory_filter_options.compute_crosshair_distance_to_engage:
        if title not in title_to_engage:
            title_to_engage[title] = []
        alive_trajectory_id_df = trajectory_id_df.iloc[:len(alive_trajectory_vis_df)]

        shot_cur_tick_series = alive_trajectory_vis_df[player_place_area_columns.player_shots_cur_tick]
        kill_next_tick_series = alive_trajectory_vis_df[player_place_area_columns.player_kill_next_tick]
        if shot_cur_tick_series.sum() == 0. and kill_next_tick_series == 0.:
            return

        time_until_next_shot = compute_time_until_next_event(alive_trajectory_id_df, shot_cur_tick_series)
        time_until_next_kill = compute_time_until_next_event(alive_trajectory_id_df, kill_next_tick_series)
        crosshair_distance_to_enemy = \
            alive_trajectory_vis_df[player_place_area_columns.nearest_crosshair_distance_to_enemy]

        result_df = pd.DataFrame({time_to_shoot_col: time_until_next_shot, time_to_kill_col: time_until_next_kill,
                                  crosshair_distance_to_enemy_col: crosshair_distance_to_enemy})





def compute_time_until_next_event(alive_trajectory_id_df: pd.DataFrame, event_series: pd.Series) -> pd.Series:
    # backwards rolling sum computes all game ticks in future where event occurs
    game_tick_if_event = alive_trajectory_id_df.loc[game_tick_number_column].where(event_series, 0)
    sum_game_tick_if_event = game_tick_if_event.iloc[::-1].cumsum().iloc[::-1]
    # remove next tick to get 
    next_shot_game_tick = sum_game_tick_if_event - sum_game_tick_if_event.shift(-1)
    ticks_until_next_event = next_shot_game_tick - alive_trajectory_id_df[game_tick_if_event]
    # filter out after last key event

    time_until_next_event = ticks_until_next_event / game_tick_rate
    time_until_next_event =
    return ticks_until_next_event / game_tick_rate
