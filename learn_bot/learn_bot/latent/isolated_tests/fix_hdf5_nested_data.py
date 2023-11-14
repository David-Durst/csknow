import h5py

from learn_bot.latent.place_area.load_data import all_train_latent_team_hdf5_dir_path

data_group_key = 'data'

hdf5_files = all_train_latent_team_hdf5_dir_path.glob('behaviorTreeTeamFeatureStore*.hdf5')
for hdf5_file_path in hdf5_files:
    with h5py.File(hdf5_file_path, 'r+') as hdf5_file:
        if data_group_key in hdf5_file[data_group_key].keys():
            print(hdf5_file_path)
            nested_hdf5_data = hdf5_file[data_group_key][data_group_key]
            for k in nested_hdf5_data.keys():
                hdf5_file.move(f'{data_group_key}/{data_group_key}/{k}', f'{data_group_key}/{k}')
            print(nested_hdf5_data.keys())
            #del nested_hdf5_data
            #print(data_group_key in hdf5_file[data_group_key].keys())
            # can't figure out how to delete groups, just move them out of the way
            print(len(hdf5_file[data_group_key].keys()))
            hdf5_file.move(f'{data_group_key}/{data_group_key}', f'old_{data_group_key}/{data_group_key}')
            print(len(hdf5_file[data_group_key].keys()))
