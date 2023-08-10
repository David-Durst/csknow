from learn_bot.latent.place_area.simulator import *
# this is a open loop version of the simulator for computing metrics based on short time horizons

num_time_steps = 10


# src tensor is variable length per round, rollout tensor is fixed length for efficiency
# fillout rollout tensor for as much as possible for each round so have ground truth for open loop predictions
def build_open_rollout_and_similarity_tensors(round_lengths: RoundLengths, dataset: LatentDataset) -> \
        Tuple[torch.Tensor,torch.Tensor]:
    rollout_tensor = torch.zeros([round_lengths.num_rounds * round_lengths.max_length_per_round, dataset.X.shape[1]])

    # get indices to copy into in rollout tensor, no indices for dataset as taking everything
    rollout_ticks_in_round = flatten_list(
        [[round_index * round_lengths.max_length_per_round + i for i in range(round_lengths.round_to_length[round_id])]
         for round_index, round_id in enumerate(round_lengths.round_ids)])
    rollout_tensor[rollout_ticks_in_round] = dataset.X

    src_first_tick_in_round = [tick_range.start for _, tick_range in round_lengths.round_to_subset_tick_indices.items()]
    similarity_tensor = dataset.similarity_tensor[src_first_tick_in_round].to(CUDA_DEVICE_STR)
    return rollout_tensor, similarity_tensor


def delta_pos_open_rollout(loaded_model: LoadedModel):
    round_lengths = get_round_lengths(loaded_model.cur_loaded_df)
    rollout_tensor, similarity_tensor = \
        build_open_rollout_and_similarity_tensors(round_lengths, loaded_model.cur_dataset)
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
