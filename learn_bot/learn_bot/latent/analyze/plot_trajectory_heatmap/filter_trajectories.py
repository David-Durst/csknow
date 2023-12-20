from dataclasses import dataclass
from typing import Set, Optional, Dict

import numpy as np
import pandas as pd

from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import AABB
from learn_bot.libs.vec import Vec3


@dataclass
class TrajectoryFilterOptions:
    valid_round_ids: Optional[Set[int]] = None
    # used in open sim to split trajectories within round (as reset after 5s) and don't want jump in discontinuity
    trajectory_counter: Optional[pd.Series] = None
    player_starts_in_region: Optional[AABB] = None
    region_name: Optional[str] = None
    include_all_players_when_one_in_region: bool = False
    round_game_seconds: Optional[range] = None
    only_kill: bool = False
    only_killed: bool = False
    only_shots: bool = False
    compute_lifetimes: bool = False
    compute_speeds: bool = False

    def __str__(self):
        compound_name = ''
        if self.player_starts_in_region is not None:
            compound_name += self.region_name.lower().replace(' ', '_') + \
                             ('_all' if self.include_all_players_when_one_in_region else '_one')
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
        if self.only_shots:
            if compound_name != '':
                compound_name += '_'
            compound_name += f'only_shots'
        if compound_name == '':
            compound_name = 'no_filter'
        return compound_name

    def filtering_key_events(self) -> bool:
        return self.only_kill or self.only_killed or self.only_shots


default_trajectory_filter_options = TrajectoryFilterOptions()

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
