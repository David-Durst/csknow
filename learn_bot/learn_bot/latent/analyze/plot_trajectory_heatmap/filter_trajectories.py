from dataclasses import dataclass
from typing import Set, Optional, Dict, List

import numpy as np
import pandas as pd

from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import AABB
from learn_bot.libs.vec import Vec3

@dataclass
class TeamBasedStartConstraint:
    name: str
    bomb_planted_a: bool
    ct_region: AABB
    t_region: AABB
    num_allowed_out_ct: int
    num_allowed_out_t: int
    # allow option to make it a requirement that at least one in one
    min_num_required_in_ct: Optional[int] = None
    min_num_required_in_t: Optional[int] = None

@dataclass
class TrajectoryFilterOptions:
    valid_round_ids: Optional[Set[int]] = None
    # used in open sim to split trajectories within round (as reset after 5s) and don't want jump in discontinuity
    trajectory_counter: Optional[pd.Series] = None
    player_starts_in_region: Optional[AABB] = None
    region_name: Optional[str] = None
    include_all_players_when_one_in_region: bool = False
    team_based_all_start_in_region: Optional[TeamBasedStartConstraint] = None
    round_game_seconds: Optional[range] = None
    only_kill: bool = False
    only_killed: bool = False
    only_killed_or_end: bool = False
    only_shots: bool = False
    compute_lifetimes: bool = False
    compute_action_changes: bool = False
    compute_speeds: bool = False
    compute_shots_per_kill: bool = False
    compute_crosshair_distance_to_engage: bool = False

    def __str__(self):
        compound_name = ''
        if self.player_starts_in_region is not None:
            compound_name += self.region_name.lower().replace(' ', '_')
            if self.include_all_players_when_one_in_region:
                compound_name += '_all'
            else:
                compound_name += '_one'
        if self.team_based_all_start_in_region is not None:
            compound_name += self.team_based_all_start_in_region.name.lower().replace(' ', '_')
        if self.round_game_seconds is not None:
            if compound_name != '':
                compound_name += '_'
            compound_name += f'time_{self.round_game_seconds.start}-{self.round_game_seconds.stop}'
        if self.only_kill:
            if compound_name != '':
                compound_name += '_'
            compound_name += f'only_kill'
        if self.only_killed:
            if compound_name != '':
                compound_name += '_'
            compound_name += f'only_killed'
        if self.only_killed_or_end:
            if compound_name != '':
                compound_name += '_'
            compound_name += f'only_killed_or_end'
        if self.only_shots:
            if compound_name != '':
                compound_name += '_'
            compound_name += f'only_shots'
        if compound_name == '':
            compound_name = 'no_filter'
        return compound_name

    def filtering_key_events(self) -> bool:
        return self.only_kill or self.only_killed or self.only_killed_or_end or self.only_shots

    def computing_multitick_metrics(self) -> bool:
        return self.compute_lifetimes or self.compute_shots_per_kill or \
            self.compute_action_changes or self.compute_crosshair_distance_to_engage

    def is_no_filter(self) -> bool:
        return str(self) == 'no_filter'


default_trajectory_filter_options = TrajectoryFilterOptions()

# specific areas
a_long_constraint = AABB(Vec3(1122, 700, 0), Vec3(1850, 1295, 0))
a_cat_constraint = AABB(Vec3(200, 1300, 0), Vec3(570, 2000, 0))
a_site_constraint = AABB(Vec3(970, 2310, 0), Vec3(1220, 2630, 0))
a_ramp_constraint = AABB(Vec3(1270, 2290, 0), Vec3(1640, 2850, 0))

mid_doors_constraint = AABB(Vec3(-600, 1670, 0), Vec3(-330, 2300, 0))

b_doors_constraint = AABB(Vec3(-1300, 2050, 0), Vec3(-1030, 2740, 0))
b_backplat_constraint = AABB(Vec3(-2150, 2450, 0), Vec3(-1720, 3180, 0))
b_site_constraint = AABB(Vec3(-1720, 2390, 0), Vec3(-1360, 2870, 0))
b_car_constraint = AABB(Vec3(-1755, 1600, 0), Vec3(-1375, 1960, 0))
b_tuns_constraint = AABB(Vec3(-2190, 960, 0), Vec3(-1500, 1435, 0))


a_long_constraint_str = 'A Long'
a_cat_constraint_str = 'A Cat'
a_site_constraint_str = 'A Site'
a_ramp_constraint_str = 'A Ramp'
mid_doors_constraint_str = 'Mid Doors'
b_doors_constraint_str = 'B Doors'
b_back_plat_constraint_str = 'B Back Plat'
b_site_constraint_str = 'B Site'
b_car_constraint_str = 'B Car'
b_tuns_constraint_str = 'B Tuns'

region_constraints: Dict[str, AABB] = {
    a_long_constraint_str: a_long_constraint,
    a_cat_constraint_str: a_cat_constraint,
    a_site_constraint_str: a_site_constraint,
    a_ramp_constraint_str: a_ramp_constraint,
    mid_doors_constraint_str: mid_doors_constraint,
    b_doors_constraint_str: b_doors_constraint,
    b_back_plat_constraint_str: b_backplat_constraint,
    b_site_constraint_str: b_site_constraint,
    b_car_constraint_str: b_car_constraint,
    b_tuns_constraint_str: b_tuns_constraint,
}

# large chunks of the map
anywhere_constraint = AABB(Vec3(-5000, -5000, -5000), Vec3(5000, 5000, 5000))
above_mid_constraint = AABB(Vec3(-5000, 2000, -5000), Vec3(5000, 5000, 5000))
from_spawn_constraint = AABB(Vec3(-296, 1983, -5000), Vec3(1267, 2800, 50))
a_not_long_constraint = AABB(Vec3(150, 2250, -5000), Vec3(5000, 5000, 5000))
a_long_constraint = AABB(Vec3(150, -5000, -5000), Vec3(5000, 2000, 5000))
c4_a_ct_spawn_t_not_long = TeamBasedStartConstraint(
    name="C4 A CT Spawn T Not Long",
    bomb_planted_a=True,
    ct_region=from_spawn_constraint,
    t_region=a_not_long_constraint,
    num_allowed_out_ct=0,
    num_allowed_out_t=0
)
c4_a_ct_spawn_t_long = TeamBasedStartConstraint(
    name="C4 A CT Spawn T Long",
    bomb_planted_a=True,
    ct_region=from_spawn_constraint,
    t_region=a_long_constraint,
    num_allowed_out_ct=0,
    num_allowed_out_t=1
)
a_not_long_not_cat_constraint = AABB(Vec3(750, 2100, -5000), Vec3(5000, 5000, 5000))
below_mid_constraint = AABB(Vec3(-5000, -5000, -5000), Vec3(5000, 625, 5000))
c4_a_ct_long_t_site = TeamBasedStartConstraint(
    name="C4 A CT Long T Site",
    bomb_planted_a=True,
    ct_region=below_mid_constraint,
    t_region=a_not_long_not_cat_constraint,
    num_allowed_out_ct=0,
    num_allowed_out_t=0
)
c4_a_ct_spawn_t_site = TeamBasedStartConstraint(
    name="C4 A CT Spawn T Site",
    bomb_planted_a=True,
    ct_region=above_mid_constraint,
    t_region=a_not_long_not_cat_constraint,
    num_allowed_out_ct=0,
    num_allowed_out_t=0
)
b_entire_site_constraint = AABB(Vec3(-5000, 1620, -5000), Vec3(-1340, 5000, 5000))
tuns_constraint = AABB(Vec3(-5000, -5000, -5000), Vec3(-673, 1600, 5000))
c4_b_ct_bslope_t_site = TeamBasedStartConstraint(
    name="C4 B CT BSlope T Site",
    bomb_planted_a=False,
    ct_region=above_mid_constraint,
    t_region=b_entire_site_constraint,
    num_allowed_out_ct=0,
    num_allowed_out_t=0
)
c4_b_ct_tuns_t_site = TeamBasedStartConstraint(
    name="C4 B CT Tuns T Site",
    bomb_planted_a=False,
    ct_region=tuns_constraint,
    t_region=b_entire_site_constraint,
    num_allowed_out_ct=0,
    num_allowed_out_t=0,
    min_num_required_in_ct=1
)
team_based_region_constraints: List[TeamBasedStartConstraint] = [
    c4_a_ct_spawn_t_not_long, c4_a_ct_spawn_t_long,
    #c4_a_ct_long_t_site, c4_a_ct_spawn_t_site,
    #c4_b_ct_bslope_t_site, c4_b_ct_tuns_t_site
]
