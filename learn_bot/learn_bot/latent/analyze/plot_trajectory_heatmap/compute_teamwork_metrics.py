from typing import List, Dict, Tuple, Union, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd

from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.order.column_names import team_strs
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.simulation.constants import place_name_to_index, place_names


class TeamPlaces:
    ct_team: bool
    planted_a: bool
    places: tuple[int]

    def __init__(self, ct_team: bool, planted_a: bool, places: Union[List[int], tuple[int], tuple[str]]):
        self.ct_team = ct_team
        self.planted_a = planted_a
        if len(places) > 0 and type(places[0]) == str:
            places = [place_name_to_index[s] for s in places]
        self.places = tuple(sorted(places))

    def __eq__(self, other) -> bool:
        return self.ct_team == other.ct_team and self.planted_a == other.planted_a and self.places == other.places

    def __hash__(self):
        return hash((self.ct_team, self.planted_a, self.places))

    def __str__(self) -> str:
        ct_str = 'CT' if self.ct_team else 'T'
        planted_str = 'A' if self.planted_a else 'B'
        return f"{ct_str} {planted_str} {','.join([place_names[p] for p in self.places])}"

    def num_players(self) -> int:
        return len(self.places)


title_to_places_to_round_counts: Dict[str, Dict[TeamPlaces, int]] = {}
title_to_places_to_tick_counts: Dict[str, Dict[TeamPlaces, int]] = {}


def print_most_common_team_places(team_place_counts: Dict[TeamPlaces, int]):
    tuple_place_count_list = [(tp, v) for tp, v in team_place_counts.items()]
    sorted_place_counts = sorted(tuple_place_count_list, key=lambda t: t[1], reverse=True)
    print('\n'.join([f"{str(tp)}: {v}" for tp, v in sorted_place_counts[:20]]))


# offense
offense_two_man_flanks_str = 'Offense Flanks'
offense_two_man_flanks = [
    TeamPlaces(True, True, ['ShortStairs', 'LongA']),
    TeamPlaces(True, True, ['ShortStairs', 'CTSpawn']),
    TeamPlaces(True, True, ['CTSpawn', 'LongA']),
    TeamPlaces(True, False, ['BDoors', 'UpperTunnel']),
    TeamPlaces(True, False, ['Hole', 'UpperTunnel']),
]
offense_two_man_executes_str = 'Offense Executes'
offense_two_man_executes = [
    TeamPlaces(True, True, ['ExtendedA', 'UnderA']),
    TeamPlaces(True, True, ['ExtendedA', 'LongA']),
    TeamPlaces(True, True, ['UnderA', 'LongA']),
    TeamPlaces(True, False, ['BombsiteB', 'Hole']),
    TeamPlaces(True, False, ['BombsiteB', 'BombsiteB']),
]

# defense
defense_default_str = 'Defense Default'
defense_default = [
    # default a
    TeamPlaces(False, True, ['BombsiteA', 'BombsiteA']),
    TeamPlaces(False, True, ['BombsiteA', 'ARamp']),
    # default B
    TeamPlaces(False, False, ['BombsiteB', 'BombsiteB']),
    TeamPlaces(False, False, ['BombsiteB', 'UpperTunnel']),
]
defense_together_off_str = 'Defense Together Off'
defense_together_off = [
    # a together but off
    TeamPlaces(False, True, ['LongA', 'LongA']),
    TeamPlaces(False, True, ['ExtendedA', 'ExtendedA']),
    # together but off
    TeamPlaces(False, False, ['BDoors', 'BDoors']),
    TeamPlaces(False, False, ['BDoors', 'Hole']),
    TeamPlaces(False, False, ['UpperTunnel', 'UpperTunnel']),
]
defense_spread_str = 'Defense Spread'
defense_spread = [
    # spread out
    TeamPlaces(False, True, ['BombsiteA', 'LongA', 'ExtendedA']),
    TeamPlaces(False, True, ['BombsiteA', 'BombsiteA', 'LongA']),
    TeamPlaces(False, True, ['BombsiteA', 'ARamp', 'LongA']),
    # spread out
    TeamPlaces(False, False, ['BombsiteB', 'BDoors', 'UpperTunnel']),
    TeamPlaces(False, False, ['BombsiteB', 'BombsiteB', 'BDoors']),
    TeamPlaces(False, False, ['BombsiteB', 'BombsiteB', 'UpperTunnel']),
]
#key_places = [
#    #TeamPlaces(True, True, ['ShortStairs', 'LongDoors', 'CTSpawn']),
#    #TeamPlaces(True, True, ['ShortStairs', 'LongDoors']),
#    #TeamPlaces(True, True, ['ShortStairs', 'ShortStairs', 'UnderA']),
#    #TeamPlaces(True, True, ['ShortStairs', 'ShortStairs', 'LongDoors']),
#    #TeamPlaces(True, True, ['ShortStairs', 'LongDoors']),
#]
all_key_places_str = 'All Places'
all_key_places = offense_two_man_flanks + offense_two_man_executes + defense_default + defense_together_off + defense_spread
grouped_key_places: Dict[str, List[TeamPlaces]] = {
    offense_two_man_flanks_str: offense_two_man_flanks,
    #offense_two_man_executes_str: offense_two_man_executes,
    #defense_default_str: defense_default,
    #defense_together_off_str: defense_together_off,
    defense_spread_str: defense_spread,
    #all_key_places_str: all_key_places
}


def get_key_place_counts(team_place_counts: Dict[TeamPlaces, int], key_places: List[TeamPlaces]) -> pd.Series:
    key_team_place_counts = {}
    for key_place in key_places:
        if key_place in team_place_counts:
            key_team_place_counts[str(key_place)] = team_place_counts[key_place]
        else:
            key_team_place_counts[str(key_place)] = 0
    return pd.Series(key_team_place_counts)


def get_key_places_by_title(key_places: List[TeamPlaces], use_tick_counts: bool) -> pd.DataFrame:
    titles: List[str] = []
    key_place_counts: List[pd.Series] = []
    counts_dict = title_to_places_to_tick_counts if use_tick_counts else title_to_places_to_round_counts
    for title, team_place_count in counts_dict.items():
        titles.append(title)
        # need to fill in empty places (one title may have them not in another) so dataframe is rectangular
        key_place_counts.append(get_key_place_counts(team_place_count, key_places))
    return pd.concat(key_place_counts, axis=1, keys=titles)

num_players_col = 'Number of Players'
ct_team_col = 'CT Team'
planted_a_col = 'Planted A'


def get_all_places_by_title() -> pd.DataFrame:
    # get all places (as some may be in one title and nother others)
    all_places: Set[TeamPlaces] = set()
    for _, team_place_count in title_to_places_to_round_counts.items():
        for team_place, round_count in team_place_count.items():
            all_places.add(team_place)
    title_to_places_str_to_round_count: Dict[str, Dict[str, int]] = {}
    for title, team_place_count in title_to_places_to_round_counts.items():
        title_to_places_str_to_round_count[title] = {}
        for place in all_places:
            if place in team_place_count:
                title_to_places_str_to_round_count[title][str(place)] = team_place_count[place]
            else:
                title_to_places_str_to_round_count[title][str(place)] = 0
    title_to_place_count_series = {title: pd.Series(place_counts) for title, place_counts in
                                   title_to_places_str_to_round_count.items()}
    places_to_num_players: Dict[str, int] = {}
    places_to_ct: Dict[str, bool] = {}
    places_to_planted_a: Dict[str, bool] = {}
    for place in all_places:
        places_to_num_players[str(place)] = place.num_players()
        places_to_ct[str(place)] = place.ct_team
        places_to_planted_a[str(place)] = place.planted_a
    title_to_place_count_series[num_players_col] = pd.Series(places_to_num_players)
    title_to_place_count_series[ct_team_col] = pd.Series(places_to_ct)
    title_to_place_count_series[planted_a_col] = pd.Series(places_to_planted_a)
    return pd.concat(title_to_place_count_series.values(), axis=1, keys=title_to_place_count_series.keys())


def print_key_team_places(team_place_counts: Dict[TeamPlaces, int]):
    for key_place in all_key_places:
        if key_place in team_place_counts:
            print(f"{str(key_place)}: {team_place_counts[key_place]}")
        else:
            print(f"{str(key_place)}: {0}")


class PlayersAliveData:
    num_ct_alive_to_num_ticks = Dict[int, int]
    num_ct_alive_to_num_rounds = Dict[int, int]
    num_t_alive_to_num_ticks = Dict[int, int]
    num_t_alive_to_num_rounds = Dict[int, int]
    num_overall_ticks = int
    num_overall_rounds = int

    def __init__(self):
        self.num_ct_alive_to_num_ticks = {}
        self.num_ct_alive_to_num_rounds = {}
        self.num_ct_alive_at_start_to_num_rounds = {}
        self.num_t_alive_to_num_ticks = {}
        self.num_t_alive_to_num_rounds = {}
        self.num_t_alive_at_start_to_num_rounds = {}
        for i in range(len(specific_player_place_area_columns) // 2):
            self.num_ct_alive_to_num_ticks[i] = 0
            self.num_ct_alive_to_num_rounds[i] = 0
            self.num_ct_alive_at_start_to_num_rounds[i] = 0
            self.num_t_alive_to_num_ticks[i] = 0
            self.num_t_alive_to_num_rounds[i] = 0
            self.num_t_alive_at_start_to_num_rounds[i] = 0
        self.num_overall_ticks = 0
        self.num_overall_rounds = 0


title_to_num_alive: Dict[str, PlayersAliveData] = {}


def get_title_to_places_to_round_counts() -> Dict[str, Dict[TeamPlaces, int]]:
    return title_to_places_to_round_counts


def get_title_to_num_alive() -> Dict[str, PlayersAliveData]:
    return title_to_num_alive


t_on_a_site_places = set([place_name_to_index[s] for s in ["BombsiteA", "ExtendedA"]])
ct_a_site_from_spawn_long = set([place_name_to_index[s] for s in ["CTSpawn", "UnderA"]])
ct_a_site_from_spawn_long_ramp = set([place_name_to_index[s] for s in ["CTSpawn", "UnderA", "ARamp"]])
ct_under_a_site_place = place_name_to_index["UnderA"]

t_away_cat_a_site_places = set([place_name_to_index[s] for s in ["BombsiteA", "ARamp", "Ramp"]])
ct_a_site_from_spawn_cat = set([place_name_to_index[s] for s in ["ExtendedA", "Short", "ShortStairs", "UnderA", "CTSpawn"]])
ct_extended_a_site_place = place_name_to_index["ExtendedA"]

t_on_b_site_places = set([place_name_to_index[s] for s in ["BombsiteB", "UpperTunnel"]])
ct_outside_b_site_place = place_name_to_index["BDoors"]

title_to_opportunities_for_a_site_mistake: Dict[str, int]
title_to_opportunities_for_a_site_round_mistake: Dict[str, int]
title_to_num_a_site_mistakes: Dict[str, int]
title_to_num_a_site_round_mistakes: Dict[str, int]
title_to_opportunities_for_b_site_mistake: Dict[str, int]
title_to_opportunities_for_b_site_round_mistake: Dict[str, int]
title_to_num_b_site_mistakes: Dict[str, int]
title_to_num_b_site_round_mistakes: Dict[str, int]


def get_title_to_opportunities_for_a_site_mistake():
    return title_to_opportunities_for_a_site_mistake


def get_title_to_opportunities_for_a_site_round_mistake():
    return title_to_opportunities_for_a_site_round_mistake


def get_title_to_num_a_site_mistakes():
    return title_to_num_a_site_mistakes


def get_title_to_num_a_site_round_mistakes():
    return title_to_num_a_site_round_mistakes


def get_title_to_opportunities_for_b_site_mistake():
    return title_to_opportunities_for_b_site_mistake


def get_title_to_opportunities_for_b_site_round_mistake():
    return title_to_opportunities_for_b_site_round_mistake


def get_title_to_num_b_site_mistakes():
    return title_to_num_b_site_mistakes


def get_title_to_num_b_site_round_mistakes():
    return title_to_num_b_site_round_mistakes


def clear_teamwork_title_caches():
    global title_to_places_to_round_counts, title_to_num_alive, \
        title_to_opportunities_for_a_site_mistake, title_to_num_a_site_mistakes, \
        title_to_opportunities_for_b_site_mistake, title_to_num_b_site_mistakes, \
        title_to_opportunities_for_a_site_round_mistake, title_to_num_a_site_round_mistakes, \
        title_to_opportunities_for_b_site_round_mistake, title_to_num_b_site_round_mistakes
    title_to_places_to_round_counts = {}
    title_to_num_alive = {}
    title_to_opportunities_for_a_site_mistake = {}
    title_to_opportunities_for_a_site_round_mistake = {}
    title_to_num_a_site_mistakes = {}
    title_to_num_a_site_round_mistakes = {}
    title_to_opportunities_for_b_site_mistake = {}
    title_to_opportunities_for_b_site_round_mistake = {}
    title_to_num_b_site_mistakes = {}
    title_to_num_b_site_round_mistakes = {}


def compute_round_metrics(loaded_model: LoadedModel, trajectory_np: np.ndarray, trajectory_vis_df: pd.DataFrame,
                             title: str):
    if title not in title_to_num_alive:
        title_to_num_alive[title] = PlayersAliveData()
    title_to_num_alive[title].num_overall_ticks += len(trajectory_np)
    title_to_num_alive[title].num_overall_rounds += 1

    planted_a_np = trajectory_np[:, loaded_model.model.c4_planted_columns[0]] > 0.5
    team_to_tick_to_place: Dict[bool, List[TeamPlaces]] = {True: [], False: []}
    for ct_team in [True, False]:
        if ct_team:
            team_range = range(0, loaded_model.model.num_players_per_team)
        else:
            team_range = range(loaded_model.model.num_players_per_team, loaded_model.model.num_players)
        alive_columns = [loaded_model.model.alive_columns[i] for i in team_range]

        place_columns = [specific_player_place_area_columns[i].place_index for i in team_range]
        places_np = trajectory_vis_df.loc[:, place_columns].to_numpy()

        round_team_places: set[TeamPlaces] = set()
        tick_team_places: Dict[TeamPlaces, int] = {}
        for i, tick_places in enumerate(places_np.tolist()):
            team_place = TeamPlaces(ct_team, planted_a_np[i],
                                             # filter out places of dead players
                                             [int(i) for i in tick_places if i < len(place_names)])
            round_team_places.add(team_place)
            if team_place not in tick_team_places:
                tick_team_places[team_place] = 0
            tick_team_places[team_place] += 1
            team_to_tick_to_place[ct_team].append(team_place)

        if title not in title_to_places_to_round_counts:
            title_to_places_to_round_counts[title] = {}
        for team_place in round_team_places:
            if team_place not in title_to_places_to_round_counts[title]:
                title_to_places_to_round_counts[title][team_place] = 0
            title_to_places_to_round_counts[title][team_place] += 1

        if title not in title_to_places_to_tick_counts:
            title_to_places_to_tick_counts[title] = {}
        for team_place in tick_team_places:
            if team_place not in title_to_places_to_tick_counts[title]:
                title_to_places_to_tick_counts[title][team_place] = 0
            title_to_places_to_tick_counts[title][team_place] += tick_team_places[team_place]

        for i in range(loaded_model.model.num_players_per_team):
            i_alive_at_start = trajectory_np[0, alive_columns].sum() == i
            if i_alive_at_start:
                if ct_team:
                    title_to_num_alive[title].num_ct_alive_at_start_to_num_rounds[i] += 1
                else:
                    title_to_num_alive[title].num_t_alive_at_start_to_num_rounds[i] += 1
            num_ticks_with_i_alive = (trajectory_np[:, alive_columns].sum(axis=1) == i).sum()
            if num_ticks_with_i_alive > 0:
                if ct_team:
                    title_to_num_alive[title].num_ct_alive_to_num_ticks[i] += num_ticks_with_i_alive
                    title_to_num_alive[title].num_ct_alive_to_num_rounds[i] += 1
                else:
                    title_to_num_alive[title].num_t_alive_to_num_ticks[i] += num_ticks_with_i_alive
                    title_to_num_alive[title].num_t_alive_to_num_rounds[i] += 1

    if title not in title_to_opportunities_for_a_site_mistake:
        title_to_opportunities_for_a_site_mistake[title] = 0
        title_to_opportunities_for_a_site_round_mistake[title] = 0

        title_to_num_a_site_mistakes[title] = 0
        title_to_num_a_site_round_mistakes[title] = 0

        title_to_opportunities_for_b_site_mistake[title] = 0
        title_to_opportunities_for_b_site_round_mistake[title] = 0

        title_to_num_b_site_mistakes[title] = 0
        title_to_num_b_site_round_mistakes[title] = 0

    had_a_round_mistake_opportunity = False
    made_a_round_mistake = False
    had_b_round_mistake_opportunity = False
    made_b_round_mistake = False
    for i in range(len(trajectory_np) - 1):
        cur_t_place = team_to_tick_to_place[False][i]
        next_t_place = team_to_tick_to_place[False][i + 1]
        cur_ct_place = team_to_tick_to_place[True][i]

        cur_t_only_on_a = set(cur_t_place.places).issubset(t_on_a_site_places)
        cur_ct_attacking_a = len(set(cur_ct_place.places).intersection(ct_a_site_from_spawn_long)) > 0
        next_t_under_a = ct_under_a_site_place in next_t_place.places

        if cur_t_only_on_a and cur_ct_attacking_a:
            title_to_opportunities_for_a_site_mistake[title] += 1
            if not had_a_round_mistake_opportunity:
                title_to_opportunities_for_a_site_round_mistake[title] += 1
            had_a_round_mistake_opportunity = True
        if cur_t_only_on_a and cur_ct_attacking_a and next_t_under_a:
            title_to_num_a_site_mistakes[title] += 1
            if not made_a_round_mistake:
                title_to_num_a_site_round_mistakes[title] += 1
            made_a_round_mistake = True

        cur_t_only_on_b = set(cur_t_place.places).issubset(t_on_b_site_places)
        next_t_outside_b = ct_outside_b_site_place in next_t_place.places
        cur_ct_outside_b = ct_outside_b_site_place in cur_ct_place.places

        if cur_t_only_on_b and cur_ct_outside_b:
            title_to_opportunities_for_b_site_mistake[title] += 1
            if not had_b_round_mistake_opportunity:
                title_to_opportunities_for_b_site_round_mistake[title] += 1
            had_b_round_mistake_opportunity = True
        if cur_t_only_on_b and next_t_outside_b and cur_ct_outside_b:
            title_to_num_b_site_mistakes[title] += 1
            if not made_b_round_mistake:
                title_to_num_b_site_round_mistakes[title] += 1
            made_b_round_mistake = True



#def compute_on_death_metrics(loaded_model: LoadedModel, trajectory_np: np.ndarray, trajectory_vis_df: pd.DataFrame,
#                             title: str):
#    killed_next_tick_columns = [player_place_area_columns.player_killed_next_tick
#                                for player_place_area_columns in specific_player_place_area_columns]
#    killed_next_tick_np = trajectory_vis_df.loc[:, killed_next_tick_columns].to_numpy()
#    ## first cumsum propagates the 1 for tick when died, the next one compounds those 1's to compute ticks since died
#    #ticks_since_killed_np = np.cumsum(np.cumsum(killed_next_tick_np > 0.5))
#    a_player_killed_next_tick_np = np.sum(killed_next_tick_np, axis=1) > 0.
#    # trajectory data when player is killed next tick
#    trajectory_where_player_killed_np = trajectory_np[a_player_killed_next_tick_np]
#    # list of actually who is killed on those ticks
#    players_killed_when_at_least_one_player_killed_np = killed_next_tick_np[a_player_killed_next_tick_np]
#
#    for row_index in range(len(trajectory_where_player_killed_np)):
#        tick_where_player_killed_np = trajectory_where_player_killed_np[row_index]
#        tick_players_killed = players_killed_when_at_least_one_player_killed_np[row_index]
#        for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
#            if not tick_players_killed[player_index]:
#                continue
#
#            # get range of teammates
#            ct_team = team_strs[0] in player_place_area_columns.player_id
#            if ct_team:
#                teammate_range = range(0, loaded_model.model.num_players_per_team)
#                enemy_range = range(loaded_model.model.num_players_per_team, loaded_model.model.num_players)
#            else:
#                teammate_range = range(loaded_model.model.num_players_per_team, loaded_model.model.num_players)
#                enemy_range = range(0, loaded_model.model.num_players_per_team)
#
#
#            # compute num teammates and enemies alive
#            teammate_alive_columns = [loaded_model.model.alive_columns[i] for i
#                                      in teammate_range
#                                      if i != player_index]
#            teammates_alive = tick_where_player_killed_np[teammate_alive_columns] > 0.5
#            num_teammates_alive = int(np.sum(tick_where_player_killed_np[teammate_alive_columns]))
#            enemy_alive_columns = [loaded_model.model.alive_columns[i] for i in enemy_range]
#            enemies_alive = tick_where_player_killed_np[enemy_alive_columns] > 0.5
#            num_enemies_alive = int(np.sum(tick_where_player_killed_np[enemy_alive_columns]))
#
#            # compute min time since seen enemy
#            teammates_seen_enemy_fov_columns = [loaded_model.model.players_visibility_fov[i] for i
#                                                in teammate_range
#                                                if i != player_index]
#            times_since_teammate_seen_enemy = tick_where_player_killed_np[teammates_seen_enemy_fov_columns]
#            min_time_since_teammate_seen_enemy = np.min(times_since_teammate_seen_enemy[teammates_alive])
#            enemies_seen_my_team_fov_columns = [loaded_model.model.players_visibility_fov[i] for i in enemy_range]
#            times_since_enemy_seen_my_team = list(tick_where_player_killed_np[enemies_seen_my_team_fov_columns][enemies_alive])
#
#            if title not in title_to_num_teammates_to_enemy_vis_on_death:
#                title_to_num_teammates_to_enemy_vis_on_death[title] = {}
#            if num_teammates_alive not in title_to_num_teammates_to_enemy_vis_on_death[title]:
#                title_to_num_teammates_to_enemy_vis_on_death[title][num_teammates_alive] = []
#            title_to_num_teammates_to_enemy_vis_on_death[title][num_teammates_alive].append(min_time_since_teammate_seen_enemy)
#
#            if title not in title_to_num_enemies_to_my_team_vis_on_death:
#                title_to_num_enemies_to_my_team_vis_on_death[title] = {}
#            if num_enemies_alive not in title_to_num_enemies_to_my_team_vis_on_death[title]:
#                title_to_num_enemies_to_my_team_vis_on_death[title][num_enemies_alive] = []
#            title_to_num_enemies_to_my_team_vis_on_death[title][num_enemies_alive] += times_since_enemy_seen_my_team
#
#            # compute distance to teammates and enemies
#            cur_player_pos = tick_where_player_killed_np[
#                loaded_model.model.nested_players_pos_columns_tensor[player_index, 0]].astype(np.float64)
#            team_distances = []
#            enemy_distances = []
#            for other_player_index, other_player_place_area_columns in enumerate(specific_player_place_area_columns):
#                # make sure not the same player and alive
#                if other_player_index == player_index:
#                    continue
#                if not tick_where_player_killed_np[loaded_model.model.alive_columns[other_player_index]]:
#                    continue
#                other_ct_team = team_strs[0] in other_player_place_area_columns.player_id
#                other_player_pos = tick_where_player_killed_np[
#                    loaded_model.model.nested_players_pos_columns_tensor[other_player_index, 0]].astype(np.float64)
#                distance = np.sum((cur_player_pos - other_player_pos) ** 2.0) ** 0.5
#                if ct_team == other_ct_team:
#                    team_distances.append(distance)
#                else:
#                    enemy_distances.append(distance)
#
#            if title not in title_to_num_teammates_to_distance_to_teammate_on_death:
#                title_to_num_teammates_to_distance_to_teammate_on_death[title] = {}
#            if num_teammates_alive not in title_to_num_teammates_to_distance_to_teammate_on_death[title]:
#                title_to_num_teammates_to_distance_to_teammate_on_death[title][num_teammates_alive] = []
#            title_to_num_teammates_to_distance_to_teammate_on_death[title][num_teammates_alive] += team_distances
#
#            if title not in title_to_num_enemies_to_distance_to_enemy_on_death:
#                title_to_num_enemies_to_distance_to_enemy_on_death[title] = {}
#            if num_enemies_alive not in title_to_num_enemies_to_distance_to_enemy_on_death[title]:
#                title_to_num_enemies_to_distance_to_enemy_on_death[title][num_enemies_alive] = []
#            title_to_num_enemies_to_distance_to_enemy_on_death[title][num_enemies_alive] += enemy_distances
#
#
#def compute_any_time_metrics(loaded_model: LoadedModel, trajectory_np: np.ndarray, trajectory_vis_df: pd.DataFrame,
#                             title: str):
#    for player_index, player_place_area_columns in enumerate(specific_player_place_area_columns):
#        ct_team = team_strs[0] in player_place_area_columns.player_id
#        cur_player_alive = trajectory_np[:, loaded_model.model.alive_columns[player_index]]
#        cur_player_pos = trajectory_np[:,
#            loaded_model.model.nested_players_pos_columns_tensor[player_index, 0]].astype(np.float64)
#        cur_player_seen_enemy_fov = trajectory_np[:, loaded_model.model.players_visibility_fov[player_index]]
#
#        ct_team = team_strs[0] in player_place_area_columns.player_id
#        if ct_team:
#            teammate_range = range(0, loaded_model.model.num_players_per_team)
#        else:
#            teammate_range = range(loaded_model.model.num_players_per_team, loaded_model.model.num_players)
#        teammate_alive_columns = [loaded_model.model.alive_columns[i] for i
#                                  in teammate_range
#                                  if i != player_index]
#        num_teammates_alive_np = trajectory_np[:, teammate_alive_columns].sum(axis=1)
#
#        for other_player_index, other_player_place_area_columns in enumerate(specific_player_place_area_columns):
#            # make sure not the same player and alive
#            if other_player_index == player_index:
#                continue
#            other_ct_team = team_strs[0] in other_player_place_area_columns.player_id
#            if other_ct_team != ct_team:
#                continue
#            other_player_alive = trajectory_np[:, loaded_model.model.alive_columns[other_player_index]]
#            other_player_pos = trajectory_np[:,
#                loaded_model.model.nested_players_pos_columns_tensor[other_player_index, 0]].astype(np.float64)
#            distance = np.sum((cur_player_pos - other_player_pos) ** 2.0, axis=1) ** 0.5
#            # not blocking if either player dead
#            both_players_alive = (cur_player_alive > 0.5) & (other_player_alive > 0.5)
#            distance[~both_players_alive] = 1000.
#            blocking_events = float(np.sum(distance < 50.))
#            # tmp cap as one round is weird
#            if blocking_events > 40.:
#                blocking_events = 40.
#
#            if blocking_events > 0.:
#                if title not in title_to_blocking_events:
#                    title_to_blocking_events[title] = []
#                title_to_blocking_events[title].append(blocking_events)
#
#            other_player_seen_enemy_fov = trajectory_np[:, loaded_model.model.players_visibility_fov[other_player_index]]
#            both_players_seen_enemy = (cur_player_seen_enemy_fov < 1.) & (other_player_seen_enemy_fov < 1.)
#            multi_engagement_distances = distance[both_players_seen_enemy & both_players_alive]
#
#            for engagement_index in range(len(multi_engagement_distances)):
#                num_teammates_alive = num_teammates_alive_np[engagement_index]
#                if title not in title_to_num_teammates_to_distance_multi_engagements:
#                    title_to_num_teammates_to_distance_multi_engagements[title] = {}
#                if num_teammates_alive not in title_to_num_teammates_to_distance_multi_engagements[title]:
#                    title_to_num_teammates_to_distance_multi_engagements[title][num_teammates_alive] = []
#                title_to_num_teammates_to_distance_multi_engagements[title][num_teammates_alive].append(multi_engagement_distances[engagement_index])
#
#            if title not in title_to_num_multi_engagements:
#                title_to_num_multi_engagements[title] = []
#            title_to_num_multi_engagements[title].append(len(multi_engagement_distances))


def compute_teamwork_metrics(loaded_model: LoadedModel, trajectory_np: np.ndarray, trajectory_vis_df: pd.DataFrame,
                             title: str):
    #compute_on_death_metrics(loaded_model, trajectory_np, trajectory_vis_df, title)
    #compute_any_time_metrics(loaded_model, trajectory_np, trajectory_vis_df, title)
    compute_round_metrics(loaded_model, trajectory_np, trajectory_vis_df, title)
