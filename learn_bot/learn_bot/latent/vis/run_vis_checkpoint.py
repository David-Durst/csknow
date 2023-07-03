from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.load_data import human_latent_team_hdf5_data_path, manual_latent_team_hdf5_data_path, \
    rollout_latent_team_hdf5_data_path, load_data
from learn_bot.libs.df_grouping import make_index_column, train_test_split_by_col_ids
from learn_bot.latent.vis.off_policy_inference import off_policy_inference
from learn_bot.latent.vis.vis import vis
from learn_bot.libs.vec import Vec3


use_manual_data = False
use_rollout_data = False
use_synthetic_data = False
use_all_human_data = True
add_manual_to_all_human_data = False
limit_manual_data_to_no_enemies_nav = True

if __name__ == "__main__":
    load_data_result = load_data(use_manual_data, use_rollout_data, use_synthetic_data, use_all_human_data,
                                 add_manual_to_all_human_data, limit_manual_data_to_no_enemies_nav)
    #if manual_data:
    #    manual_data = HDF5Wrapper(manual_latent_team_hdf5_data_path, hdf5_id_columns)
    #    all_data_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=[i for i in range(20000)])
    #    #all_data_df = all_data_df[all_data_df['test name'] == b'LearnedGooseToCatScript']
    #elif rollout_data:
    #    all_data_df = load_hdf5_to_pd(rollout_latent_team_hdf5_data_path)
    #else:
    #    all_data_df = load_hdf5_to_pd(human_latent_team_hdf5_data_path)
    #all_data_df = all_data_df.copy()

    #all_data_df = filter_region(all_data_df, AABB(Vec3(-580., 1740., 0.), Vec3(-280., 2088., 0.)), True, False,
    #                            [1, 2, 3, 4])

    #for flip_column in [ColumnsToFlip(" CT 1", " CT 2")]:
    #    flip_column.apply_flip(all_data_df)

    loaded_model = load_model_file(load_data_result)
    vis(loaded_model, off_policy_inference)
