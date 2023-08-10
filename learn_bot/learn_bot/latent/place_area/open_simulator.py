from learn_bot.latent.place_area.simulator import *
# this is a open loop version of the simulator for computing metrics based on short time horizons

num_time_steps = 10


def delta_pos_open_rollout(loaded_model: LoadedModel):
    round_lengths = get_round_lengths(loaded_model.cur_loaded_df)
    rollout_tensor, similarity_tensor = \
        build_rollout_and_similarity_tensors(round_lengths, loaded_model.cur_dataset)
    pred_tensor = torch.zeros(rollout_tensor.shape[0], loaded_model.cur_dataset.Y.shape[1])
    loaded_model.model.eval()
    with torch.no_grad():
        num_steps = round_lengths.max_length_per_round - 1
        with tqdm(total=num_steps, disable=False) as pbar:
            for step_index in range(num_steps):
                if (step_index + 1) % num_time_steps != 0:
                    step(rollout_tensor, similarity_tensor, pred_tensor, loaded_model.model, round_lengths, step_index, nav_data)
                pbar.update(1)
    # need to modify cur_loaded_df as rollout_df has constant length of all rounds for sim efficiency
    loaded_model.cur_loaded_df, loaded_model.cur_inference_df = \
        match_round_lengths(loaded_model.cur_loaded_df, rollout_tensor, pred_tensor, round_lengths,
                            loaded_model.column_transformers)


nav_data = None

if __name__ == "__main__":
    nav_data = NavData(CUDA_DEVICE_STR)

    load_data_result = LoadDataResult(load_data_options)
    #if manual_data:
    #    all_data_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=[i for i in range(20000)])
    #    #all_data_df = all_data_df[all_data_df['test name'] == b'LearnedGooseToCatScript']
    #elif rollout_data:
    #    all_data_df = load_hdf5_to_pd(rollout_latent_team_hdf5_data_path)
    #else:
    #    all_data_df = load_hdf5_to_pd(human_latent_team_hdf5_data_path, rows_to_get=[i for i in range(20000)])
    #all_data_df = all_data_df.copy()

    #load_result = load_model_file_for_rollout(all_data_df, "delta_pos_checkpoint.pt")

    loaded_model = load_model_file(load_data_result, use_test_data_only=True)
    vis(loaded_model, delta_pos_open_rollout)
