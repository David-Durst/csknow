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

delta_distance_to_nearest_teammate_name = "delta distance to nearest teammate"
delta_distance_to_nearest_teammate_when_firing_name = "delta distance to nearest teammate when firing"
delta_distance_to_nearest_teammate_when_shot_name = "delta distance to nearest teammate when shot"

distance_to_nearest_enemy_name = "distance to nearest enemy"
distance_to_nearest_enemy_when_firing_name = "distance to nearest enemy when firing"
distance_to_nearest_enemy_when_shot_name = "distance to nearest enemy when shot"

distance_to_attacker_when_shot_name = "distance to attacker when shot"

distance_to_cover_name = "distance to cover"
distance_to_cover_when_enemy_visible_no_fov_name = "distance to cover when enemy visible no fov"
distance_to_cover_when_enemy_visible_fov_name = "distance to cover when enemy visible fov"
distance_to_cover_when_firing_name = "distance to cover when firing"
distance_to_cover_when_shot_name = "distance to cover when shot"

distance_to_c4_name = "distance to cover"
distance_to_c4_when_enemy_visible_fov_name = "distance to cover when enemy visible fov"
distance_to_c4_when_firing_name = "distance to cover when firing"
distance_to_c4_when_shot_name = "distance to cover when shot"

delta_distance_to_c4_name = "delta distance to cover"
delta_distance_to_c4_when_enemy_visible_fov_name = "delta distance to cover when enemy visible fov"
delta_distance_to_c4_when_firing_name = "delta distance to cover when firing"
delta_distance_to_c4_when_shot_name = "delta distance to cover when shot"

time_from_firing_to_teammate_seeing_enemy_fov_name = "time from firing to teammate seeing enemy fov"
time_from_shot_to_teammate_seeing_enemy_fov_name = "time from shot to teammate seeing enemy fov"

pct_time_max_speed_ct_name = "pct time max speed ct"
pct_time_max_speed_t_name = "pct time max speed t"
pct_time_still_ct_name = "pct time still ct"
pct_time_still_t_name = "pct time still t"
ct_wins_name = "ct wins"
ct_wins_title = "Offense Wins"

round_id_per_pat_name = "round id per pat"
round_id_per_firing_pat_name = "round id per firing pat"
round_id_per_shot_pat_name = "round id per shot pat"
round_id_per_nearest_teammate_name = "round id per nearest teammate"
round_id_per_nearest_teammate_firing_name = "round id per nearest teammate firing"
round_id_per_nearest_teammate_shot_name = "round id per nearest teammate shot"
round_id_per_enemy_visible_no_fov_pat_name = "round id per enemy visible no fov pat"
round_id_per_enemy_visible_fov_pat_name = "round id per enemy visible fov pat"
round_id_per_firing_to_teammate_seeing_enemy_name = "round id per firing to teammate seeing enemy"
round_id_per_shot_to_teammate_seeing_enemy_name = "round id per shot to teammate seeing enemy"

is_ct_per_pat_name = "is ct per pat"
is_ct_per_firing_pat_name = "is ct per firing pat"
is_ct_per_shot_pat_name = "is ct per shot pat"
is_ct_per_nearest_teammate_name = "is ct per nearest teammate"
is_ct_per_nearest_teammate_firing_name = "is ct per nearest teammate firing"
is_ct_per_nearest_teammate_shot_name = "is ct per nearest teammate shot"
is_ct_per_enemy_visible_no_fov_pat_name = "is ct per enemy visible no fov pat"
is_ct_per_enemy_visible_fov_pat_name = "is ct per enemy visible fov pat"
is_ct_per_firing_to_teammate_seeing_enemy_name = "is ct per firing to teammate seeing enemy"
is_ct_per_shot_to_teammate_seeing_enemy_name = "is ct per shot to teammate seeing enemy"

player_id_per_pat_name = "player id per pat"
player_id_per_firing_pat_name = "player id per firing pat"
player_id_per_shot_pat_name = "player id per shot pat"
player_id_per_nearest_teammate_name = "player id per nearest teammate"
player_id_per_nearest_teammate_firing_name = "player id per nearest teammate firing"
player_id_per_nearest_teammate_shot_name = "player id per nearest teammate shot"
player_id_per_enemy_visible_no_fov_pat_name = "player id per enemy visible no fov pat"
player_id_per_enemy_visible_fov_pat_name = "player id per enemy visible fov pat"
player_id_per_firing_to_teammate_seeing_enemy_name = "player id per firing to teammate seeing enemy"
player_id_per_shot_to_teammate_seeing_enemy_name = "player id per shot to teammate seeing enemy"

round_id_per_round_name = "round id per round"

rollout_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics.hdf5'
heuristic_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics_heuristic.hdf5'
default_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics_default.hdf5'
all_train_humanness_folder_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs'

all_train_humanness_hdf5_data_paths: List[Path] = []

for p in all_train_humanness_folder_path.glob('humannessMetrics*.hdf5'):
    all_train_humanness_hdf5_data_paths.append(p)



