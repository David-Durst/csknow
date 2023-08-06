from pathlib import Path
from typing import List

velocity_when_firing = "velocity when firing"
velocity_when_shot = "velocity when shot"
distance_to_nearest_teammate_when_firing = "distance to nearest teammate when firing"
distance_to_nearest_enemy_when_firing = "distance to nearest enemy when firing"
distance_to_nearest_teammate_when_shot = "distance to nearest teammate when shot"
distance_to_attacker_when_shot = "distance to attacker when shot"
distance_to_cover_when_firing = "distance to cover when firing"
distance_to_cover_when_shot = "distance to cover when shot"
pct_time_max_speed_ct = "pct time max speed ct"
pct_time_max_speed_t = "pct time max speed t"
pct_time_still_ct = "pct time still ct"
pct_time_still_t = "pct time still t"
ct_wins = "ct wins"

rollout_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics.hdf5'
all_train_humanness_folder_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs'

all_train_humanness_hdf5_data_paths: List[Path] = []

for p in all_train_humanness_folder_path.glob('humannessMetrics*.hdf5'):
    all_train_humanness_hdf5_data_paths.append(p)



