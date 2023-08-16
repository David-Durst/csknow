from pathlib import Path
from typing import List

humanness_plots_path = Path(__file__).parent / 'humanness_plots'

unscaled_speed_name = "unscaled speed"
unscaled_speed_when_firing_name = "unscaled speed when firing"
unscaled_speed_when_shot_name = "unscaled speed when shot"

scaled_speed_name = "scaled speed"
scaled_speed_when_firing_name = "scaled speed when firing"
scaled_speed_when_shot_name = "scaled speed when shot"

weapon_only_scaled_speed_name = "weapon only scaled speed"
weapon_only_scaled_speed_when_firing_name = "weapon only scaled speed when firing"
weapon_only_scaled_speed_when_shot_name = "weapon only scaled speed when shot"

distance_to_nearest_teammate_name = "distance to nearest teammate"
distance_to_nearest_teammate_when_firing_name = "distance to nearest teammate when firing"
distance_to_nearest_teammate_when_shot_name = "distance to nearest teammate when shot"

distance_to_nearest_enemy_name = "distance to nearest enemy"
distance_to_nearest_enemy_when_firing_name = "distance to nearest enemy when firing"
distance_to_nearest_enemy_when_shot_name = "distance to nearest enemy when shot"

distance_to_attacker_when_shot_name = "distance to attacker when shot"

distance_to_cover_name = "distance to cover"
distance_to_cover_when_enemy_visible_no_fov_name = "distance to cover when enemy visible no fov"
distance_to_cover_when_enemy_visible_fov_name = "distance to cover when enemy visible fov"
distance_to_cover_when_firing_name = "distance to cover when firing"
distance_to_cover_when_shot_name = "distance to cover when shot"

pct_time_max_speed_ct_name = "pct time max speed ct"
pct_time_max_speed_t_name = "pct time max speed t"
pct_time_still_ct_name = "pct time still ct"
pct_time_still_t_name = "pct time still t"
ct_wins_name = "ct wins"
ct_wins_title = "Offense Wins"

rollout_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics.hdf5'
heuristic_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics_heuristic.hdf5'
default_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics_default.hdf5'
all_train_humanness_folder_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs'

all_train_humanness_hdf5_data_paths: List[Path] = []

for p in all_train_humanness_folder_path.glob('humannessMetrics*.hdf5'):
    all_train_humanness_hdf5_data_paths.append(p)



