import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import pandas as pd
from PIL import Image, ImageFont

from learn_bot.latent.analyze.compare_trajectories.filter_trajectory_key_events import FilterEventType, \
    KeyAreas, KeyAreaTeam
from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_and_events import FilterPlayerType, \
    plot_trajectory_dfs_and_event
from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import set_pd_print_options, \
    ComparisonConfig
from learn_bot.latent.analyze.comparison_column_names import metric_type_col, best_fit_ground_truth_trace_batch_col, predicted_trace_batch_col, best_match_id_col, predicted_round_id_col, \
    best_fit_ground_truth_round_id_col, predicted_round_number_col, best_fit_ground_truth_round_number_col
from learn_bot.latent.engagement.column_names import round_id_column, round_number_column
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import AABB
from learn_bot.libs.vec import Vec3

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


key_areas: KeyAreas = [
    # cat
    AABB(Vec3(245., 2000., -50), Vec3(510., 2070., 10000)),
    # longa
    AABB(Vec3(1160., 1130., -10000), Vec3(1640., 1300., 10000)),
    # underA
    AABB(Vec3(1010., 2020., -10000), Vec3(1210., 2300., 10000)),
    # bdoors
    AABB(Vec3(-1450., 2030., -10000), Vec3(-1000., 2890., 10000)),
    # btuns
    AABB(Vec3(-2100., 1220., -10000), Vec3(-1880., 1550., 10000))
]
key_area_names = ['ACat', 'LongA', 'UnderA', 'BDoors', 'BTuns']
key_area_team = ['']

class TrajectoryPlots:
    unfiltered: Image
    filtered_fire: Image
    filtered_kill: Image
    filtered_areas: List[Image.Image]
    filtered_fire_and_areas: List[Image.Image]


def plot_predicted_trajectory_per_team(predicted_trajectory_dfs: List[pd.DataFrame], config: ComparisonConfig,
                                       similarity_plots_path: Path,
                                       include_ct: bool, include_t: bool) -> TrajectoryPlots:
    use_team_str = f" ct: {include_ct}, t: {include_t}"
    team_file_ending = ".png"
    if include_t and not include_ct:
        team_file_ending = "_t.png"
    elif not include_t and include_ct:
        team_file_ending = "_ct.png"

    result = TrajectoryPlots()

    print("plotting predicted" + use_team_str)
    result.unfiltered = plot_trajectory_dfs_and_event(predicted_trajectory_dfs, config, True, include_ct, include_t)
    #result.unfiltered.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories' + team_file_ending))

    print("plotting predicted fire events" + use_team_str)
    result.filtered_fire = plot_trajectory_dfs_and_event(predicted_trajectory_dfs, config, True, include_ct, include_t,
                                                         FilterPlayerType.IncludeOnlyInEvent,
                                                         FilterEventType.Fire)
    #result.filtered_fire.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_fire' +
    #                                                   team_file_ending))

    print("plotting predicted kill events" + use_team_str)
    result.filtered_kill = plot_trajectory_dfs_and_event(predicted_trajectory_dfs, config, True, include_ct, include_t,
                                                         FilterPlayerType.IncludeOnlyInEvent,
                                                         FilterEventType.Kill)
    #result.filtered_kill.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_kill' +
    #                                                   team_file_ending))

    result.filtered_areas = []
    result.filtered_fire_and_areas = []
    for i in range(len(key_areas)):
        title_appendix = " " + key_area_names[i]
        print("plotting predicted key area events" + title_appendix + use_team_str)
        result.filtered_areas.append(plot_trajectory_dfs_and_event(predicted_trajectory_dfs, config, True, include_ct, include_t,
                                                                   FilterPlayerType.ExcludeOneInEvent,
                                                                   FilterEventType.KeyArea, [key_areas[i]], KeyAreaTeam.CT,
                                                                   title_appendix=title_appendix))
        result.filtered_areas[-1].save(similarity_plots_path / (config.metric_cost_file_name +
                                                                '_trajectories_area_' + key_area_names[i] +
                                                                team_file_ending))

        print("plotting predicted fire and key area events" + title_appendix + use_team_str)
        result.filtered_fire_and_areas.append(plot_trajectory_dfs_and_event(predicted_trajectory_dfs, config, True,
                                                                            include_ct, include_t,
                                                                            FilterPlayerType.ExcludeOneInEvent,
                                                                            FilterEventType.FireAndKeyArea,
                                                                            [key_areas[i]], KeyAreaTeam.CT,
                                                                            title_appendix=title_appendix))
        result.filtered_fire_and_areas[-1].save(similarity_plots_path /
                                                (config.metric_cost_file_name + '_trajectories_fire_and_area_' +
                                                 key_area_names[i] + team_file_ending))

    return result


def concat_horizontal(ims: List[Image.Image]) -> Image.Image:
    dst = Image.new('RGB', (sum([im.width for im in ims]), ims[0].height))
    offset = 0
    for im in ims:
        dst.paste(im, (offset, 0))
        offset += im.width
    return dst


def concat_vertical(ims: List[Image.Image]) -> Image.Image:
    dst = Image.new('RGB', (ims[0].width, sum([im.height for im in ims])))
    offset = 0
    for im in ims:
        dst.paste(im, (0, offset))
        offset += im.height
    return dst


def concat_trajectory_plots(trajectory_plots: List[TrajectoryPlots], similarity_plots_path: Path,
                            config: ComparisonConfig) -> TrajectoryPlots:
    result = TrajectoryPlots()
    result.unfiltered = concat_horizontal([t.unfiltered for t in trajectory_plots])
    #result.unfiltered.save(similarity_plots_path / (config.metric_cost_file_name + '_trajectories_all_teams.png'))
    result.filtered_fire = concat_horizontal([t.filtered_fire for t in trajectory_plots])
    #result.filtered_fire.save(similarity_plots_path / (config.metric_cost_file_name + '_fire_all_teams.png'))
    result.filtered_kill = concat_horizontal([t.filtered_kill for t in trajectory_plots])
    #result.filtered_kill.save(similarity_plots_path / (config.metric_cost_file_name + '_kill_all_teams.png'))
    result.filtered_areas = []
    result.filtered_fire_and_areas = []
    for i in range(len(trajectory_plots[0].filtered_areas)):
        result.filtered_areas.append(concat_horizontal([t.filtered_areas[i] for t in trajectory_plots]))
        #result.filtered_areas[-1].save(similarity_plots_path / (config.metric_cost_file_name + '_area_' +
        #                                                        key_area_names[i] + '.png'))
        result.filtered_fire_and_areas.append(concat_horizontal([t.filtered_fire_and_areas[i] for t in trajectory_plots]))
        #result.filtered_fire_and_areas[-1].save(similarity_plots_path / (config.metric_cost_file_name +
        #                                                                 '_fire_and_area_' +
        #                                                                 key_area_names[i] + '.png'))
    return result


def concat_trajectory_plots_across_player_type(trajectory_plots: List[TrajectoryPlots], similarity_plots_path: Path):
    result = TrajectoryPlots()
    result.unfiltered = concat_vertical([t.unfiltered for t in trajectory_plots])
    result.unfiltered.save(similarity_plots_path / 'all_player_types_trajectories_all_teams.png')
    result.filtered_fire = concat_vertical([t.filtered_fire for t in trajectory_plots])
    result.filtered_fire.save(similarity_plots_path / 'all_player_types_trajectories_fire_all_teams.png')
    result.filtered_kill = concat_vertical([t.filtered_kill for t in trajectory_plots])
    result.filtered_kill.save(similarity_plots_path / 'all_player_types_trajectories_kill_all_teams.png')
    result.filtered_areas = []
    result.filtered_fire_and_areas = []
    for i in range(len(trajectory_plots[0].filtered_areas)):
        result.filtered_areas.append(concat_vertical([t.filtered_areas[i] for t in trajectory_plots]))
        result.filtered_areas[-1].save(similarity_plots_path / ('all_player_types_trajectories_area_' +
                                                                key_area_names[i] + '.png'))
        result.filtered_fire_and_areas.append(concat_vertical([t.filtered_fire_and_areas[i] for t in trajectory_plots]))
        result.filtered_fire_and_areas[-1].save(similarity_plots_path / ('all_player_types_trajectories_fire_and_area_'
                                                                         + key_area_names[i] + '.png'))


debug_caching = True
remove_debug_cache = False
plot_ground_truth = False


def plot_trajectory_comparisons(similarity_df: pd.DataFrame, predicted_loaded_model: LoadedModel,
                                ground_truth_loaded_model: LoadedModel,
                                config: ComparisonConfig, similarity_plots_path: Path,
                                debug_caching_override: bool = False) -> TrajectoryPlots:
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
    debug_cache_path = similarity_plots_path / "predicted_trajectory_dfs.pickle"
    if debug_caching and debug_caching_override and debug_cache_path.is_file():
        with open(debug_cache_path, "rb") as pickle_file:
            predicted_trajectory_dfs = pickle.load(pickle_file)
    else:
        predicted_trajectory_dfs = select_trajectories_into_dfs(predicted_loaded_model,
                                                                list(predicted_rounds_for_comparison_heatmap))
        if debug_caching and debug_caching_override:
            with open(debug_cache_path, "wb") as pickle_file:
                pickle.dump(predicted_trajectory_dfs, pickle_file)
        elif remove_debug_cache:
            debug_cache_path.unlink(missing_ok=True)

    if plot_ground_truth:
        print('loading best fit ground truth df')
        best_fit_ground_truth_trajectory_dfs = \
            select_trajectories_into_dfs(ground_truth_loaded_model,
                                         list(best_fit_ground_truth_rounds_for_comparison_heatmap))

    both_teams_predicted_images = \
        plot_predicted_trajectory_per_team(predicted_trajectory_dfs, config, similarity_plots_path, True, True)
    ct_predicted_images = \
        plot_predicted_trajectory_per_team(predicted_trajectory_dfs, config, similarity_plots_path, True, False)
    t_predicted_images = \
        plot_predicted_trajectory_per_team(predicted_trajectory_dfs, config, similarity_plots_path, False, True)
    return concat_trajectory_plots([both_teams_predicted_images, ct_predicted_images, t_predicted_images],
                                   similarity_plots_path, config)


    #if plot_ground_truth:
    #    assert False
    #    ground_truth_image = plot_trajectory_dfs(best_fit_ground_truth_trajectory_dfs, config, False, True, True)

    #    combined_image = Image.new('RGB', (predicted_image.width + ground_truth_image.width, predicted_image.height))
    #    combined_image.paste(predicted_image, (0, 0))
    #    combined_image.paste(ground_truth_image, (predicted_image.width, 0))
    #    combined_image.save(similarity_plots_path / (config.metric_cost_file_name + '_similarity_trajectories' + '.png'))

    #print("plotting teammate")
    #plot_occupancy_heatmap(predicted_trajectory_dfs, config, distance_to_other_player=True,
    #                       teammate=True, similarity_plots_path=similarity_plots_path)
    #print("plotting enemy")
    #plot_occupancy_heatmap(predicted_trajectory_dfs, config, distance_to_other_player=True,
    #                       teammate=False, similarity_plots_path=similarity_plots_path)
