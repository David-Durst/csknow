import h5py

from learn_bot.latent.analyze.process_trajectory_comparison import set_pd_print_options
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

f_path = "/home/durst/dev/csknow/demo_parser/hdf5/all_train_data/22.hdf5"
#f_path = "/home/durst/dev/csknow/analytics/all_train_outputs/behaviorTreeTeamFeatureStore_28.hdf5"

with h5py.File(f_path) as hdf5_file:
    print(hdf5_file.keys())
    print('hi')