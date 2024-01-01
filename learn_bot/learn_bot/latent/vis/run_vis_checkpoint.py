from learn_bot.latent.analyze.comparison_column_names import small_human_good_rounds, \
    all_human_vs_small_human_similarity_hdf5_data_path, all_human_28_second_filter_good_rounds, \
    all_human_vs_human_28_similarity_hdf5_data_path
from learn_bot.latent.load_model import load_model_file, LoadedModel
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
    #hand_labeled_push_round_ids=[all_human_28_second_filter_good_rounds],
    #similarity_dfs=[load_hdf5_to_pd(all_human_vs_human_28_similarity_hdf5_data_path)],
    load_similarity_data=True,
    limit_by_similarity=False,
    train_test_split_file_name=train_test_split_file_name
)

rollout_load_data_options = LoadDataOptions(
    use_manual_data=False,
    use_rollout_data=True,
    use_synthetic_data=False,
    use_small_human_data=False,
    use_human_28_data=False,
    use_all_human_data=False,
    add_manual_to_all_human_data=False,
    limit_manual_data_to_no_enemies_nav=False,
    custom_rollout_extension='_12_29_23_model_learned_no_time_with_partial_1474_rounds*'

)

def compute_num_points(load_data_result: LoadDataResult):
    all_points = 0
    train_points = 0
    test_points = 0
    for hdf5_wrapper in load_data_result.multi_hdf5_wrapper.hdf5_wrappers:
        all_points += len(hdf5_wrapper.id_df)
    for hdf5_wrapper in load_data_result.multi_hdf5_wrapper.train_hdf5_wrappers:
        train_points += len(hdf5_wrapper.id_df)
    for hdf5_wrapper in load_data_result.multi_hdf5_wrapper.test_hdf5_wrappers:
        test_points += len(hdf5_wrapper.id_df)
    print(f"all points in data set: {all_points}, train points {train_points}, test points {test_points}")


if __name__ == "__main__":
    load_data_result = LoadDataResult(load_data_options)
    compute_num_points(load_data_result)

    #loaded_model = load_model_file(load_data_result, use_test_data_only=True)
    loaded_model = load_model_file(load_data_result)

    vis(loaded_model, off_policy_inference, " Off Policy")
