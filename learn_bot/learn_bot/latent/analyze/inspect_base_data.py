import h5py

from learn_bot.latent.analyze.process_trajectory_comparison import set_pd_print_options
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

with h5py.File("/home/durst/dev/csknow/demo_parser/hdf5/all_train_data/12.hdf5") as hdf5_file:
    print(hdf5_file.keys())
    print('hi')