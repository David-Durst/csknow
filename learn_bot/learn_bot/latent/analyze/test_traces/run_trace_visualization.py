import os
from typing import Optional

import pandas as pd

from learn_bot.latent.analyze.test_traces.plot_trace_humanness_metrics import plot_humanness_metrics, \
    RoundBotPlayerIds, TraceBotPlayerIds
from learn_bot.latent.analyze.test_traces.run_trace_creation import *
from learn_bot.latent.analyze.test_traces.column_names import *
from PIL import Image, ImageDraw

from learn_bot.libs.multi_hdf5_wrapper import train_test_split_folder_path
from learn_bot.mining.area_cluster import d2_radar_path
from learn_bot.latent.vis.draw_inference import minimapWidth, minimapHeight, d2_top_left_x, d2_top_left_y, minimapScale

d2_img = Image.open(d2_radar_path)
d2_img = d2_img.resize((minimapWidth, minimapHeight), Image.Resampling.LANCZOS)

bot_ct_color_list = [4, 190, 196, 0]
replay_ct_color_list = [18, 237, 147, 0]
bot_t_color_list = [187, 142, 52, 0]
replay_t_color_list = [237, 18, 108, 0]


def convert_to_canvas_coordinates(x_coords: pd.Series, y_coords: pd.Series) -> Tuple[pd.Series, pd.Series]:
    return (x_coords - d2_top_left_x) / minimapScale, (d2_top_left_y - y_coords) / minimapScale


def draw_trace_paths(trace_df: pd.DataFrame, trace_extra_df: pd.DataFrame, trace_index: int, one_non_replay_bot: bool,
                     trace_style_appendix: str, all_human: bool = False, trace_demo_file: Optional[str] = None,
                     cur_trace_round_ids: Optional[List[int]] = None) -> RoundBotPlayerIds:
    if trace_demo_file is None or len(cur_trace_round_ids) is None:
        cur_trace_extra_df = trace_extra_df[(trace_extra_df[trace_index_name] == trace_index) &
                                            (trace_extra_df[trace_one_non_replay_bot_name] == one_non_replay_bot)]
        cur_trace_round_ids = cur_trace_extra_df.index
        trace_demo_file = cur_trace_extra_df.loc[cur_trace_round_ids[0], trace_demo_file_name].decode('utf-8')[:-1]
        if len(cur_trace_round_ids) != 10:
            print('too few round ids')
    num_trace_repeats = len(cur_trace_round_ids)
    # set alpha so blend to 255 if fully overlap
    bot_ct_color = (bot_ct_color_list[0], bot_ct_color_list[1], bot_ct_color_list[2], 255 // num_trace_repeats)
    replay_ct_color = \
        (replay_ct_color_list[0], replay_ct_color_list[1], replay_ct_color_list[2], 255 // num_trace_repeats)
    bot_t_color = (bot_t_color_list[0], bot_t_color_list[1], bot_t_color_list[2], 255 // num_trace_repeats)
    replay_t_color = \
        (replay_t_color_list[0], replay_t_color_list[1], replay_t_color_list[2], 255 // num_trace_repeats)
    start_color = (255, 255, 255, 255)

    all_player_d2_img_copy = d2_img.copy().convert("RGBA")

    round_bot_player_ids: RoundBotPlayerIds = {}

    first_round_in_trace = True
    for cur_round_id in cur_trace_round_ids:
        # since this was split with : rather than _, need to remove last _
        cur_round_trace_df = trace_df[trace_df[round_id_column] == cur_round_id]
        round_bot_player_ids[cur_round_id] = []
        for player_place_area_columns in specific_player_place_area_columns:
            cur_round_player_trace_df = cur_round_trace_df[cur_round_trace_df[player_place_area_columns.alive] == 1]
            if cur_round_player_trace_df.empty:
                continue
            player_x_coords = cur_round_player_trace_df.loc[:, player_place_area_columns.pos[0]]
            player_y_coords = cur_round_player_trace_df.loc[:, player_place_area_columns.pos[1]]
            player_canvas_x_coords, player_canvas_y_coords = \
                convert_to_canvas_coordinates(player_x_coords, player_y_coords)
            player_xy_coords = list(zip(list(player_canvas_x_coords), list(player_canvas_y_coords)))

            cur_player_d2_overlay_im = Image.new("RGBA", all_player_d2_img_copy.size, (255, 255, 255, 0))
            cur_player_d2_drw = ImageDraw.Draw(cur_player_d2_overlay_im)

            ct_team = team_strs[0] in player_place_area_columns.player_id

            is_bot = False if all_human else cur_trace_extra_df.loc[cur_round_id, player_place_area_columns.trace_is_bot_player]
            if is_bot:
                round_bot_player_ids[cur_round_id].append(
                    cur_round_player_trace_df[player_place_area_columns.player_id].iloc[0])
            if ct_team:
                if is_bot:
                    fill_color = bot_ct_color
                else:
                    fill_color = replay_ct_color
            else:
                if is_bot:
                    fill_color = bot_t_color
                else:
                    fill_color = replay_t_color
            cur_player_d2_drw.line(xy=player_xy_coords, fill=fill_color, width=5)
            if first_round_in_trace:
                start_xy = player_xy_coords[0]
                cur_player_d2_drw.rectangle((start_xy[0] - 5, start_xy[1] - 5, start_xy[0] + 5, start_xy[1] + 5),
                                            fill=start_color)
            all_player_d2_img_copy.alpha_composite(cur_player_d2_overlay_im)
        first_round_in_trace = False

    png_file_name = str(trace_index) + "_" + trace_demo_file + "_" + str(one_non_replay_bot) + ".png"
    os.makedirs(trace_plots_path / trace_style_appendix, exist_ok=True)
    all_player_d2_img_copy.save(trace_plots_path / trace_style_appendix / png_file_name)
    print(f"finished {png_file_name}")

    return round_bot_player_ids


def visualize_bot_traces(trace_hdf5_data_path: Path, trace_style_appendix: str) -> TraceBotPlayerIds:
    trace_df = load_hdf5_to_pd(trace_hdf5_data_path)
    trace_extra_df = load_hdf5_to_pd(trace_hdf5_data_path, root_key='extra',
                                     cols_to_get=[trace_demo_file_name, trace_index_name, num_traces_name,
                                                  trace_one_non_replay_team_name, trace_one_non_replay_bot_name] + trace_is_bot_player_names)

    trace_bot_player_ids: TraceBotPlayerIds = {}

    for trace_index in range(len(rounds_for_traces)):
        one_team_player_ids = draw_trace_paths(trace_df, trace_extra_df, trace_index, False, trace_style_appendix)
        one_bot_player_ids = draw_trace_paths(trace_df, trace_extra_df, trace_index, True, trace_style_appendix)
        trace_bot_player_ids[trace_index] = {**one_bot_player_ids, **one_team_player_ids}

    return trace_bot_player_ids


def visualize_human_traces():
    trace_path = train_test_split_folder_path / trace_file_name
    trace_df = load_hdf5_to_pd(trace_path)
    trace_extra_df = load_hdf5_to_pd(trace_path, root_key='extra',
                                     cols_to_get=[rft_demo_file_name, rft_start_index_in_hdf5_name, rft_length_name,
                                                  rft_hdf5_key])

    for trace_index, extra_row in trace_extra_df.iterrows():
        start_index = extra_row[rft_start_index_in_hdf5_name]
        end_index = start_index + extra_row[rft_length_name]
        single_trace_df = trace_df.iloc[start_index:end_index, :]
        round_id = single_trace_df[round_id_column].iloc[0]
        draw_trace_paths(single_trace_df, pd.DataFrame(), trace_index, False, 'human', True,
                         extra_row[rft_demo_file_name].decode('utf-8'), [round_id])


if __name__ == "__main__":
    #set_pd_print_options()

    #visualize_human_traces()
    aggressive_trace_bot_player_ids = visualize_bot_traces(rollout_aggressive_trace_hdf5_data_path, 'aggressive')
    passive_trace_bot_player_ids = visualize_bot_traces(rollout_passive_trace_hdf5_data_path, 'passive')
    heuristic_trace_bot_player_ids = visualize_bot_traces(rollout_heuristic_trace_hdf5_data_path, 'heuristic')
    default_trace_bot_player_ids = visualize_bot_traces(rollout_default_trace_hdf5_data_path, 'default')

    plot_humanness_metrics(aggressive_trace_bot_player_ids, passive_trace_bot_player_ids)
