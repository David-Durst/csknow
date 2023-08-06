from pathlib import Path
from typing import List

humanness_plots_path = Path(__file__).parent / 'humanness_plots'

velocity_when_firing_name = "velocity when firing"
velocity_when_shot_name = "velocity when shot"
distance_to_nearest_teammate_when_firing_name = "distance to nearest teammate when firing"
distance_to_nearest_enemy_when_firing_name = "distance to nearest enemy when firing"
distance_to_nearest_teammate_when_shot_name = "distance to nearest teammate when shot"
distance_to_attacker_when_shot_name = "distance to attacker when shot"
distance_to_cover_when_firing_name = "distance to cover when firing"
distance_to_cover_when_shot_name = "distance to cover when shot"
pct_time_max_speed_ct_name = "pct time max speed ct"
pct_time_max_speed_t_name = "pct time max speed t"
pct_time_still_ct_name = "pct time still ct"
pct_time_still_t_name = "pct time still t"
ct_wins_name = "ct wins"

rollout_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics.hdf5'
all_train_humanness_folder_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs'

all_train_humanness_hdf5_data_paths: List[Path] = []

for p in all_train_humanness_folder_path.glob('humannessMetrics*.hdf5'):
    all_train_humanness_hdf5_data_paths.append(p)



