from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import set_pd_print_options, \
    ComparisonConfig
from learn_bot.latent.analyze.comparison_column_names import metric_type_col, dtw_cost_col, \
    best_fit_ground_truth_trace_batch_col, predicted_trace_batch_col, best_match_id_col, predicted_round_id_col, \
    best_fit_ground_truth_round_id_col, predicted_round_number_col, best_fit_ground_truth_round_number_col
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, convert_to_canvas_coordinates, \
    bot_ct_color_list, bot_t_color_list
from learn_bot.latent.engagement.column_names import round_id_column, round_number_column
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns

plot_n_most_similar = 1


@dataclass(frozen=True, eq=True)
class RoundForComparisonHeatmap:
    hdf5_file_name: str
    round_id: int
    round_number: int


def select_trajectories_into_dfs(loaded_model: LoadedModel,
                                 rounds_for_comparison: List[RoundForComparisonHeatmap]) -> List[pd.DataFrame]:
    selected_dfs: List[pd.DataFrame] = []
    hdf5s_for_selection: Set[str] = {r.hdf5_file_name for r in rounds_for_comparison}

    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        print(f"Processing hdf5 {i + 1} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
        loaded_model.cur_hdf5_index = i
        wrapper_hdf5_file_name = hdf5_wrapper.hdf5_path.name

        if wrapper_hdf5_file_name not in hdf5s_for_selection:
            continue

        loaded_model.load_cur_hdf5_as_pd(load_cur_dataset=False, cast_bool_to_int=False)
        for round_for_comparison in rounds_for_comparison:
            if wrapper_hdf5_file_name == round_for_comparison.hdf5_file_name:
                selected_dfs.append(loaded_model.cur_loaded_df[loaded_model.cur_loaded_df[round_id_column] ==
                                                               round_for_comparison.round_id])
                assert selected_dfs[-1][round_number_column].iloc[0] == round_for_comparison.round_number

    return selected_dfs


title_font = ImageFont.truetype("arial.ttf", 25)


def plot_trajectory_dfs(trajectory_dfs: List[pd.DataFrame], config: ComparisonConfig, predicted: bool) -> Image:

    all_player_d2_img_copy = d2_img.copy().convert("RGBA")
    # ground truth has many copies, scale it's color down so brightness comparable
    color_alpha = 20 #* plot_n_most_similar if predicted else 20
    ct_color = (bot_ct_color_list[0], bot_ct_color_list[1], bot_ct_color_list[2], color_alpha)
    t_color = (bot_t_color_list[0], bot_t_color_list[1], bot_t_color_list[2], color_alpha)

    for trajectory_df in trajectory_dfs:
        # since this was split with : rather than _, need to remove last _
        for player_place_area_columns in specific_player_place_area_columns:
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
            # drawing same text over and over again, so no big deal
            title_text = config.metric_cost_title + " " + ("Predicted" if predicted else "Ground Truth")
            _, _, w, h = cur_player_d2_drw.textbbox((0, 0), title_text, font=title_font)
            cur_player_d2_drw.text(((all_player_d2_img_copy.width - w) / 2,
                                    (all_player_d2_img_copy.height * 0.1 - h) / 2),
                                   title_text, fill=(255, 255, 255, 255), font=title_font)

            ct_team = team_strs[0] in player_place_area_columns.player_id
            if ct_team:
                fill_color = ct_color
            else:
                fill_color = t_color
            cur_player_d2_drw.line(xy=player_xy_coords, fill=fill_color, width=5)
            all_player_d2_img_copy.alpha_composite(cur_player_d2_overlay_im)

    return all_player_d2_img_copy


def plot_trajectory_comparison_heatmaps(similarity_df: pd.DataFrame, predicted_loaded_model: LoadedModel,
                                        ground_truth_loaded_model: LoadedModel,
                                        config: ComparisonConfig, similarity_plots_path: Path):
    set_pd_print_options()
    # print(similarity_df.loc[:, [predicted_round_id_col, best_fit_ground_truth_round_id_col, metric_type_col,
    #                            dtw_cost_col, delta_distance_col, delta_time_col]])

    # plot cost, distance, and time by metric type
    metric_types = similarity_df[metric_type_col].unique().tolist()
    metric_types = [m.decode('utf-8') for m in metric_types]
    slope_metric_type = [m for m in metric_types if 'Slope' in m][0]
    relevant_similarity_df = similarity_df[(similarity_df[metric_type_col].str.decode('utf-8') == slope_metric_type) &
                                           (similarity_df[best_match_id_col] < plot_n_most_similar)]

    # same predicted will have multiple matches, start with set and convert to list
    predicted_rounds_for_comparison_heatmap: Set[RoundForComparisonHeatmap] = set()
    best_fit_ground_truth_rounds_for_comparison_heatmap: Set[RoundForComparisonHeatmap] = set()
    for _, similarity_row in relevant_similarity_df.iterrows():
        predicted_rounds_for_comparison_heatmap.add(
            RoundForComparisonHeatmap(similarity_row[predicted_trace_batch_col].decode('utf-8'),
                                  similarity_row[predicted_round_id_col],
                                  similarity_row[predicted_round_number_col])
        )
        best_fit_ground_truth_rounds_for_comparison_heatmap.add(
            RoundForComparisonHeatmap(similarity_row[best_fit_ground_truth_trace_batch_col].decode('utf-8'),
                                      similarity_row[best_fit_ground_truth_round_id_col],
                                      similarity_row[best_fit_ground_truth_round_number_col])
        )

    print('loading predicted df')
    predicted_trajectory_dfs = select_trajectories_into_dfs(predicted_loaded_model,
                                                            list(predicted_rounds_for_comparison_heatmap))
    predicted_image = plot_trajectory_dfs(predicted_trajectory_dfs, config, True)
    print('loading best fit ground truth df')
    best_fit_ground_truth_trajectory_dfs = \
        select_trajectories_into_dfs(ground_truth_loaded_model,
                                     list(best_fit_ground_truth_rounds_for_comparison_heatmap))
    ground_truth_image = plot_trajectory_dfs(best_fit_ground_truth_trajectory_dfs, config, False)

    combined_image = Image.new('RGB', (predicted_image.width + ground_truth_image.width, predicted_image.height))
    combined_image.paste(predicted_image, (0, 0))
    combined_image.paste(ground_truth_image, (predicted_image.width, 0))
    combined_image.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories' + '.png'))