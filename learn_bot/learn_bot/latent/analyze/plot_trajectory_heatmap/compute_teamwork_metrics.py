from typing import List, Dict

import numpy as np
import pandas as pd

from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns

title_to_num_teammates_to_enemy_vis_on_death: Dict[str, Dict[int, List[float]]] = {}
title_to_num_teammates_to_distance_to_teammate_on_death: Dict[str, Dict[int, List[float]]] = {}
title_to_num_enemies_to_my_team_vis_on_death: Dict[str, Dict[int, List[float]]] = {}
title_to_num_enemies_to_distance_to_enemy_on_death: Dict[str, Dict[int, List[float]]] = {}
title_to_blocking_events: Dict[str, List[float]] = {}


def get_title_to_num_teammates_to_enemy_vis_on_death() -> Dict[str, Dict[int, List[float]]]:
    return title_to_num_teammates_to_enemy_vis_on_death


def get_title_to_num_enemies_to_my_team_vis_on_death() -> Dict[str, Dict[int, List[float]]]:
    return title_to_num_enemies_to_my_team_vis_on_death


def get_title_to_num_teammates_to_distance_to_teammate_on_death() -> Dict[str, Dict[int, List[float]]]:
    return title_to_num_teammates_to_distance_to_teammate_on_death


def get_title_to_num_enemies_to_distance_to_enemy_on_death() -> Dict[str, Dict[int, List[float]]]:
    return title_to_num_enemies_to_distance_to_enemy_on_death


def get_title_to_blocking_events() -> Dict[str, List[float]]:
    return title_to_blocking_events


def clear_teamwork_title_caches():
    global title_to_num_teammates_to_enemy_vis_on_death, title_to_num_enemies_to_my_team_vis_on_death, \
        title_to_num_teammates_to_distance_to_teammate_on_death, title_to_num_enemies_to_distance_to_enemy_on_death, \
        title_to_blocking_events
    title_to_num_teammates_to_enemy_vis_on_death = {}
    title_to_num_enemies_to_my_team_vis_on_death = {}
    title_to_num_teammates_to_distance_to_teammate_on_death = {}
    title_to_num_enemies_to_distance_to_enemy_on_death = {}
    title_to_blocking_events = {}


def compute_on_death_metrics(loaded_model: LoadedModel, trajectory_np: np.ndarray, trajectory_vis_df: pd.DataFrame,
                             title: str):
    killed_next_tick_columns = [player_place_area_columns.player_killed_next_tick
                                for player_place_area_columns in specific_player_place_area_columns]
    killed_next_tick_np = trajectory_vis_df.loc[:, killed_next_tick_columns].to_numpy()
    ## first cumsum propagates the 1 for tick when died, the next one compounds those 1's to compute ticks since died
    #ticks_since_killed_np = np.cumsum(np.cumsum(killed_next_tick_np > 0.5))
    a_player_killed_next_tick_np = np.sum(killed_next_tick_np, axis=1) > 0.
    # trajectory data when player is killed next tick
    trajectory_where_player_killed_np = trajectory_np[a_player_killed_next_tick_np]
    # list of actually who is killed on those ticks
    players_killed_when_at_least_one_player_killed_np = killed_next_tick_np[a_player_killed_next_tick_np]

    for row_index in range(len(trajectory_where_player_killed_np)):
        tick_where_player_killed_np = trajectory_where_player_killed_np[row_index]
        tick_players_killed = players_killed_when_at_least_one_player_killed_np[row_index]
        for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
            if not tick_players_killed[player_index]:
                continue

            # get range of teammates
            ct_team = team_strs[0] in player_place_area_columns.player_id
            if ct_team:
                teammate_range = range(0, loaded_model.model.num_players_per_team)
                enemy_range = range(loaded_model.model.num_players_per_team, loaded_model.model.num_players)
            else:
                teammate_range = range(loaded_model.model.num_players_per_team, loaded_model.model.num_players)
                enemy_range = range(0, loaded_model.model.num_players_per_team)


            # compute num teammates and enemies alive
            teammate_alive_columns = [loaded_model.model.alive_columns[i] for i
                                      in teammate_range
                                      if i != player_index]
            teammates_alive = tick_where_player_killed_np[teammate_alive_columns] > 0.5
            num_teammates_alive = int(np.sum(tick_where_player_killed_np[teammate_alive_columns]))
            enemy_alive_columns = [loaded_model.model.alive_columns[i] for i in enemy_range]
            enemies_alive = tick_where_player_killed_np[enemy_alive_columns] > 0.5
            num_enemies_alive = int(np.sum(tick_where_player_killed_np[enemy_alive_columns]))

            # compute min time since seen enemy
            teammates_seen_enemy_fov_columns = [loaded_model.model.players_visibility_fov[i] for i
                                                in teammate_range
                                                if i != player_index]
            times_since_teammate_seen_enemy = tick_where_player_killed_np[teammates_seen_enemy_fov_columns]
            min_time_since_teammate_seen_enemy = np.min(times_since_teammate_seen_enemy[teammates_alive])
            enemies_seen_my_team_fov_columns = [loaded_model.model.players_visibility_fov[i] for i in enemy_range]
            times_since_enemy_seen_my_team = list(tick_where_player_killed_np[enemies_seen_my_team_fov_columns][enemies_alive])

            if title not in title_to_num_teammates_to_enemy_vis_on_death:
                title_to_num_teammates_to_enemy_vis_on_death[title] = {}
            if num_teammates_alive not in title_to_num_teammates_to_enemy_vis_on_death[title]:
                title_to_num_teammates_to_enemy_vis_on_death[title][num_teammates_alive] = []
            title_to_num_teammates_to_enemy_vis_on_death[title][num_teammates_alive].append(min_time_since_teammate_seen_enemy)

            if title not in title_to_num_enemies_to_my_team_vis_on_death:
                title_to_num_enemies_to_my_team_vis_on_death[title] = {}
            if num_enemies_alive not in title_to_num_enemies_to_my_team_vis_on_death[title]:
                title_to_num_enemies_to_my_team_vis_on_death[title][num_enemies_alive] = []
            title_to_num_enemies_to_my_team_vis_on_death[title][num_enemies_alive] += times_since_enemy_seen_my_team

            # compute distance to teammates and enemies
            cur_player_pos = tick_where_player_killed_np[
                loaded_model.model.nested_players_pos_columns_tensor[player_index, 0]].astype(np.float64)
            team_distances = []
            enemy_distances = []
            for other_player_index, other_player_place_area_columns in enumerate(specific_player_place_area_columns):
                # make sure not the same player and alive
                if other_player_index == player_index:
                    continue
                if not tick_where_player_killed_np[loaded_model.model.alive_columns[other_player_index]]:
                    continue
                other_ct_team = team_strs[0] in other_player_place_area_columns.player_id
                other_player_pos = tick_where_player_killed_np[
                    loaded_model.model.nested_players_pos_columns_tensor[other_player_index, 0]].astype(np.float64)
                distance = np.sum((cur_player_pos - other_player_pos) ** 2.0) ** 0.5
                if ct_team == other_ct_team:
                    team_distances.append(distance)
                else:
                    enemy_distances.append(distance)

            if title not in title_to_num_teammates_to_distance_to_teammate_on_death:
                title_to_num_teammates_to_distance_to_teammate_on_death[title] = {}
            if num_teammates_alive not in title_to_num_teammates_to_distance_to_teammate_on_death[title]:
                title_to_num_teammates_to_distance_to_teammate_on_death[title][num_teammates_alive] = []
            title_to_num_teammates_to_distance_to_teammate_on_death[title][num_teammates_alive] += team_distances

            if title not in title_to_num_enemies_to_distance_to_enemy_on_death:
                title_to_num_enemies_to_distance_to_enemy_on_death[title] = {}
            if num_enemies_alive not in title_to_num_enemies_to_distance_to_enemy_on_death[title]:
                title_to_num_enemies_to_distance_to_enemy_on_death[title][num_enemies_alive] = []
            title_to_num_enemies_to_distance_to_enemy_on_death[title][num_enemies_alive] += enemy_distances


def compute_any_time_metrics(loaded_model: LoadedModel, trajectory_np: np.ndarray, trajectory_vis_df: pd.DataFrame,
                             title: str):
    for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
        ct_team = team_strs[0] in player_place_area_columns.player_id
        cur_player_alive = trajectory_np[:, loaded_model.model.alive_columns[player_index]]
        cur_player_pos = trajectory_np[:,
            loaded_model.model.nested_players_pos_columns_tensor[player_index, 0]].astype(np.float64)
        for other_player_index, other_player_place_area_columns in enumerate(specific_player_place_area_columns):
            # make sure not the same player and alive
            if other_player_index == player_index:
                continue
            other_ct_team = team_strs[0] in other_player_place_area_columns.player_id
            if other_ct_team != ct_team:
                continue
            other_player_alive = trajectory_np[:, loaded_model.model.alive_columns[other_player_index]]
            other_player_pos = trajectory_np[:,
                loaded_model.model.nested_players_pos_columns_tensor[other_player_index, 0]].astype(np.float64)
            distance = np.sum((cur_player_pos - other_player_pos) ** 2.0, axis=1) ** 0.5
            # not blocking if either player dead
            distance[~((cur_player_alive > 0.5) & (other_player_alive > 0.5))] = 1000.
            blocking_events = float(np.sum(distance < 50.))
            # tmp cap as one round is weird
            if blocking_events > 40.:
                blocking_events = 40.

            if blocking_events > 0.:
                if title not in title_to_blocking_events:
                    title_to_blocking_events[title] = []
                title_to_blocking_events[title].append(blocking_events)


def compute_teamwork_metrics(loaded_model: LoadedModel, trajectory_np: np.ndarray, trajectory_vis_df: pd.DataFrame,
                             title: str):
    compute_on_death_metrics(loaded_model, trajectory_np, trajectory_vis_df, title)
    compute_any_time_metrics(loaded_model, trajectory_np, trajectory_vis_df, title)
