from learn_bot.latent.analyze.comparison_column_names import small_human_good_rounds, \
    all_human_vs_small_human_similarity_hdf5_data_path, all_human_28_second_filter_good_rounds, \
    all_human_vs_human_28_similarity_hdf5_data_path
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.load_data import LoadDataResult, LoadDataOptions
from learn_bot.latent.train_paths import train_test_split_file_name
from learn_bot.latent.vis.off_policy_inference import off_policy_inference
from learn_bot.latent.vis.vis import vis
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

load_data_options = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=False,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_human_28_data=False,
    use_all_human_data=True,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=False,
    limit_manual_data_to_only_enemies_no_nav=False,
    small_good_rounds=[all_human_28_second_filter_good_rounds],
    similarity_dfs=[load_hdf5_to_pd(all_human_vs_human_28_similarity_hdf5_data_path)],
    limit_by_similarity=False,
    train_test_split_file_name=train_test_split_file_name
)

if __name__ == "__main__":
    load_data_result = LoadDataResult(load_data_options)
    #load_data_result.multi_hdf5_wrapper.hdf5_wrappers[0].limit(load_data_result.multi_hdf5_wrapper.hdf5_wrappers[0].id_df['round id'] < 700)

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

    #set_pd_print_options()
    #non_delta_pos_cols = []
    #for col in load_data_result.multi_hdf5_wrapper.hdf5_wrappers[0].sample_df.columns:
    #    if "delta pos" not in col:
    #        non_delta_pos_cols.append(col)
    #data_series = load_data_result.multi_hdf5_wrapper.hdf5_wrappers[0].sample_df.iloc[0].loc[non_delta_pos_cols]
    #print(data_series)
    #loaded_model = load_model_file(load_data_result, use_test_data_only=True)
    loaded_model = load_model_file(load_data_result)
    vis(loaded_model, off_policy_inference, " Off Policy")
