import tempfile
from enum import Enum
from math import log, ceil
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.filter_trajectory_key_events import FilterEventType, KeyAreas, \
    KeyAreaTeam, filter_trajectory_by_key_events
from learn_bot.latent.analyze.compare_trajectories.plot_distance_to_other_player import extra_data_from_metric_title, \
    plot_occupancy_heatmap
from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import ComparisonConfig, generate_bins, \
    plot_hist
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, bot_ct_color_list, bot_t_color_list, \
    convert_to_canvas_coordinates, replay_ct_color_list, replay_t_color_list
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns, delta_pos_grid_cell_dim
from learn_bot.latent.vis.draw_inference import VisMapCoordinate

title_font = ImageFont.truetype("arial.ttf", 25)

class FilterPlayerType(Enum):
    IncludeAll = 1
    IncludeOnlyInEvent = 2
    ExcludeOneInEvent = 3


def plot_trajectory_dfs_and_event(trajectory_dfs: List[pd.DataFrame], config: ComparisonConfig, predicted: bool,
                                  include_ct: bool, include_t: bool,
                                  filter_players: Optional[FilterPlayerType] = FilterPlayerType.IncludeAll,
                                  filter_event_type: Optional[FilterEventType] = None,
                                  key_areas: Optional[KeyAreas] = None, key_area_team: KeyAreaTeam = KeyAreaTeam.Both,
                                  title_appendix: str = "",
                                  plot_starts: bool = False,
                                  # if this is set, only plot post start for this player in each trajectory
                                  only_plot_post_start: Optional[List[int]] = None) -> Image.Image:
    filtered_trajectory_dfs: List[pd.DataFrame] = []
    valid_players_dfs: List[pd.DataFrame] = []
    num_points = 0
    if filter_event_type is not None:
        for round_trajectory_df in trajectory_dfs:
            # split round trajectory by filter
            cur_round_filtered_trajectory_dfs = \
                filter_trajectory_by_key_events(filter_event_type, round_trajectory_df,
                                                filter_players != FilterPlayerType.IncludeOnlyInEvent,
                                                key_areas, key_area_team)
            for data_df in cur_round_filtered_trajectory_dfs.data:
                filtered_trajectory_dfs.append(data_df)
                num_points += len(data_df)
            for valid_players_df in cur_round_filtered_trajectory_dfs.valid_players:
                valid_players_dfs.append(valid_players_df)
    else:
        filtered_trajectory_dfs = trajectory_dfs
        valid_players_dfs = [pd.DataFrame() for _ in trajectory_dfs]
        for df in trajectory_dfs:
            num_points += len(df)

    if filter_players == FilterPlayerType.IncludeOnlyInEvent:
        return plot_events(filtered_trajectory_dfs, valid_players_dfs,
                           config, predicted, include_ct, include_t,
                           filter_players, filter_event_type, title_appendix)
    else:
        return plot_trajectory_dfs(filtered_trajectory_dfs, valid_players_dfs, num_points, len(trajectory_dfs),
                                   config, predicted, include_ct, include_t,
                                   filter_players,
                                   filter_event_type,
                                   title_appendix,
                                   plot_starts=plot_starts,
                                   only_plot_post_start=only_plot_post_start)


#def plot_speeds(trajectory_dfs: List[pd.DataFrame], valid_players_dfs: List[pd.DataFrame],
#               config: ComparisonConfig, predicted: bool, include_ct: bool, include_t: bool,
#               filter_event_type: Optional[FilterEventType] = None,
#               title_appendix: str = "") -> Image.Image:
#    trajectory_df = pd.concat(trajectory_dfs)
#    valid_players_df = pd.concat(valid_players_dfs)
#
#    speeds: List[pd.Series] = []
#    for player_place_area_columns in specific_player_place_area_columns:
#        ct_team = team_strs[0] in player_place_area_columns.player_id
#        if ct_team and not include_ct:
#            continue
#        elif not ct_team and not include_t:
#            continue
#
#        # make sure player is alive if don't have another condition
#        if len(valid_players_df) == 0:
#            cur_player_trajectory_df = trajectory_df[trajectory_df[player_place_area_columns.alive] == 1]
#            if cur_player_trajectory_df.empty:
#                continue
#        else:
#            cur_player_trajectory_df = trajectory_df[valid_players_df[player_place_area_columns.player_id]]
#
#        speeds.append((cur_player_trajectory_df[player_place_area_columns.vel[0]] ** 2 +
#                       cur_player_trajectory_df[player_place_area_columns.vel[1]] ** 2 +
#                       cur_player_trajectory_df[player_place_area_columns.vel[2]] ** 2) ** (1. / 2.))
#
#    speed = pd.concat(speeds)
#    bins = generate_bins(0, 300, 300 // 20)
#
#    title_text = get_title(config, filter_event_type, include_ct, include_t, predicted, title_appendix)
#    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
#    fig.suptitle(title_text, fontsize=16)
#    ax = fig.subplots(1, 1)
#
#    plot_hist(ax, speed, bins)
#    ax.text(150, 0.4, speed.describe().to_string(), family='monospace')
#    ax.set_ylim(0., 1.)
#
#    tmp_dir = tempfile.gettempdir()
#    tmp_file = Path(tmp_dir) / 'tmp_heatmap.png'
#    plt.savefig(tmp_file)
#    img = Image.open(tmp_file)
#    img.load()
#    tmp_file.unlink()
#    return img


def plot_events(filtered_trajectory_dfs: List[pd.DataFrame], valid_players_dfs: List[pd.DataFrame],
                config: ComparisonConfig, predicted: bool, include_ct: bool, include_t: bool,
                filter_players: Optional[FilterPlayerType] = FilterPlayerType.IncludeAll,
                filter_event_type: Optional[FilterEventType] = None,
                title_appendix: str = "") -> Image.Image:
    assert filter_players == FilterPlayerType.IncludeOnlyInEvent
    title_text = get_title(config, filter_event_type, include_ct, include_t, predicted, title_appendix)
    return plot_occupancy_heatmap(filtered_trajectory_dfs, config, False, False, None, valid_players_dfs,
                                  include_ct=include_ct, include_t=include_t,
                                  title_text=title_text)


def get_title(config, filter_event_type, include_ct, include_t, predicted, title_appendix):
    team_text = f" CT: {include_ct}, T: {include_t}"
    event_text = "" if filter_event_type is None else (" " + str(filter_event_type))
    title_text = extra_data_from_metric_title(config.metric_cost_title, predicted) + \
                 event_text + team_text + title_appendix
    return title_text


def plot_trajectory_dfs(filtered_trajectory_dfs: List[pd.DataFrame], valid_players_dfs: List[pd.DataFrame],
                        num_points: int, num_unfiltered_trajectories: int, config: ComparisonConfig,
                        predicted: bool, include_ct: bool, include_t: bool,
                        filter_players: Optional[FilterPlayerType] = FilterPlayerType.IncludeAll,
                        filter_event_type: Optional[FilterEventType] = None,
                        title_appendix: str = "",
                        plot_starts: bool = False,
                        # if this is set, only plot post start for this player in each trajectory
                        only_plot_post_start: Optional[List[int]] = None) -> Image.Image:
    all_player_d2_img_copy = d2_img.copy().convert("RGBA")
    # if only looking at one player, scale up alpha as not drawing a lot of ponts
    if only_plot_post_start is not None:
        num_points /= 4.
    color_alpha = int(25. / log(2.2 + num_points / 1300, 10))
    ct_color = (bot_ct_color_list[0], bot_ct_color_list[1], bot_ct_color_list[2], color_alpha)
    start_ct_color = (replay_ct_color_list[0], replay_ct_color_list[1], replay_ct_color_list[2], color_alpha)
    t_color = (bot_t_color_list[0], bot_t_color_list[1], bot_t_color_list[2], color_alpha)
    start_t_color = (replay_t_color_list[0], replay_t_color_list[1], replay_t_color_list[2], color_alpha)

    first_title = True
    team_text = f" CT: {include_ct}, T: {include_t}"
    event_text = "" if filter_event_type is None else (" " + str(filter_event_type))
    points_text = f"\nNum Points {num_points} Alpha {color_alpha}"

    with tqdm(total=len(filtered_trajectory_dfs), disable=False) as pbar:
        trajectory_index = -1
        for trajectory_df, valid_players_df in zip(filtered_trajectory_dfs, valid_players_dfs):
            assert filter_players != FilterPlayerType.IncludeOnlyInEvent
            trajectory_index += 1

            # filter out the player who appears in event the most
            player_ids_to_exclude: Set[int] = set()
            if filter_players == FilterPlayerType.ExcludeOneInEvent:
                max_player_id_col_name = None
                max_player_valids = 0
                for player_id_col_name in valid_players_df.columns:
                    cur_player_valids = sum(valid_players_df[player_id_col_name])
                    if cur_player_valids > max_player_valids:
                        max_player_id_col_name = player_id_col_name
                        max_player_valids = cur_player_valids
                player_ids_to_exclude.add(max_player_id_col_name)

            for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
                if player_place_area_columns.player_id in player_ids_to_exclude:
                    continue

                ct_team = team_strs[0] in player_place_area_columns.player_id
                if ct_team:
                    if not include_ct:
                        continue
                    fill_color = ct_color
                    start_fill_color = start_ct_color
                else:
                    if not include_t:
                        continue
                    fill_color = t_color
                    start_fill_color = start_t_color

                cur_player_trajectory_df = trajectory_df[trajectory_df[player_place_area_columns.alive] == 1]
                if cur_player_trajectory_df.empty:
                    continue
                player_x_coords = cur_player_trajectory_df.loc[:, player_place_area_columns.pos[0]]
                player_y_coords = cur_player_trajectory_df.loc[:, player_place_area_columns.pos[1]]
                player_canvas_x_coords, player_canvas_y_coords = \
                    convert_to_canvas_coordinates(player_x_coords, player_y_coords)
                player_xy_coords = list(zip(list(player_canvas_x_coords), list(player_canvas_y_coords)))

                cur_player_d2_overlay_im = Image.new("RGBA", all_player_d2_img_copy.size, (255, 255, 255, 0))
                cur_player_d2_drw = ImageDraw.Draw(cur_player_d2_overlay_im)
                if first_title:
                    title_text = extra_data_from_metric_title(config.metric_cost_title, predicted) + \
                                 event_text + team_text + title_appendix + points_text
                    _, _, w, h = cur_player_d2_drw.textbbox((0, 0), title_text, font=title_font)
                    cur_player_d2_drw.text(((all_player_d2_img_copy.width - w) / 2,
                                            (all_player_d2_img_copy.height * 0.1 - h) / 2),
                                           title_text, fill=(255, 255, 255, 255), font=title_font)
                    first_title = False

                if only_plot_post_start is None or player_index == only_plot_post_start[trajectory_index]:
                    #if player_y_coords[0] < 1900:
                    #    print(cur_player_trajectory_df['round id'])
                    #    print(cur_player_trajectory_df['hdf5 id'])
                    #    exit(0)
                    cur_player_d2_drw.line(xy=player_xy_coords, fill=fill_color, width=5)
                if plot_starts:
                    cur_player_d2_drw.rectangle(
                        (player_xy_coords[0][0] - 5, player_xy_coords[0][1] - 5,
                         player_xy_coords[0][0] + 5, player_xy_coords[0][1] + 5),
                        fill=start_fill_color)
                all_player_d2_img_copy.alpha_composite(cur_player_d2_overlay_im)
            pbar.update(1)

    print(f"num trajectories in plot {num_unfiltered_trajectories}, alpha {color_alpha}, num points {num_points}")

    #heatmap = plot_occupancy_heatmap(filtered_trajectory_dfs, config, distance_to_other_player=False,
    #                                 teammate=False, similarity_plots_path=None)

    return all_player_d2_img_copy#LineAndHeatmapPlots(all_player_d2_img_copy, heatmap)


