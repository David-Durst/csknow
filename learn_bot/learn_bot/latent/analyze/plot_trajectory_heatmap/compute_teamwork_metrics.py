from typing import List, Dict

import numpy as np
import pandas as pd

from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns

title_to_num_teammates_to_enemy_vis_on_death: Dict[str, Dict[int, List[float]]] = {}
title_to_num_enemies_to_my_team_vis_on_death: Dict[str, Dict[int, List[float]]] = {}


def get_title_to_num_teammates_to_enemy_vis_on_death() -> Dict[str, Dict[int, List[float]]]:
    return title_to_num_teammates_to_enemy_vis_on_death


def get_title_to_num_enemies_to_my_team_vis_on_death() -> Dict[str, Dict[int, List[float]]]:
    return title_to_num_enemies_to_my_team_vis_on_death


def clear_teamwork_title_caches():
    global title_to_num_teammates_to_enemy_vis_on_death, title_to_num_enemies_to_my_team_vis_on_death
    title_to_num_teammates_to_enemy_vis_on_death = {}
    title_to_num_enemies_to_my_team_vis_on_death = {}


def compute_teamwork_metrics(loaded_model: LoadedModel, trajectory_np: np.ndarray, trajectory_vis_df: pd.DataFrame,
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

            # compute min time since seen enemy
            teammates_seen_enemy_fov_columns = [loaded_model.model.players_visibility_fov[i] for i
                                                in teammate_range
                                                if i != player_index]
            min_time_since_teammate_seen_enemy = np.min(tick_where_player_killed_np[teammates_seen_enemy_fov_columns])
            enemies_seen_my_team_fov_columns = [loaded_model.model.players_visibility_fov[i] for i in enemy_range]
            min_time_since_enemy_seen_my_team = np.min(tick_where_player_killed_np[enemies_seen_my_team_fov_columns])

            # compute num teammates and enemies alive
            teammate_alive_columns = [loaded_model.model.alive_columns[i] for i
                                      in teammate_range
                                      if i != player_index]
            num_teammates_alive = int(np.sum(tick_where_player_killed_np[teammate_alive_columns]))
            enemy_alive_columns = [loaded_model.model.alive_columns[i] for i in enemy_range]
            num_enemies_alive = int(np.sum(tick_where_player_killed_np[enemy_alive_columns]))

            if title not in title_to_num_teammates_to_enemy_vis_on_death:
                title_to_num_teammates_to_enemy_vis_on_death[title] = {}
            if num_teammates_alive not in title_to_num_teammates_to_enemy_vis_on_death[title]:
                title_to_num_teammates_to_enemy_vis_on_death[title][num_teammates_alive] = []
            title_to_num_teammates_to_enemy_vis_on_death[title][num_teammates_alive].append(min_time_since_teammate_seen_enemy)

            if title not in title_to_num_enemies_to_my_team_vis_on_death:
                title_to_num_enemies_to_my_team_vis_on_death[title] = {}
            if num_enemies_alive not in title_to_num_enemies_to_my_team_vis_on_death[title]:
                title_to_num_enemies_to_my_team_vis_on_death[title][num_enemies_alive] = []
            title_to_num_enemies_to_my_team_vis_on_death[title][num_enemies_alive].append(min_time_since_enemy_seen_my_team)

