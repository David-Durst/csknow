from learn_bot.latent.engagement.column_names import game_id_column, round_number_column
from learn_bot.latent.place_area.simulation.simulator import *
from dataclasses import dataclass

from learn_bot.libs.hdf5_to_pd import save_pd_to_hdf5
from learn_bot.libs.multi_hdf5_wrapper import absolute_to_relative_train_test_key


rft_demo_file_name = "demo file"
rft_round_number_name = "round number"
rft_ct_bot_name = "ct bot"
rft_one_bot_feature_store_index_name = "one bot feature store index"
rft_start_index_in_hdf5_name = "start index in hdf5"
rft_length_name = "length"
rft_hdf5_key = "hdf5 key"


@dataclass
class RoundForTrace:
    demo_file: str
    round_number: int
    ct_bot: bool
    one_bot_feature_store_index: int
    start_index_in_hdf5: int = -1
    length: int = -1
    hdf5_key: str = ""
    round_id: int = -1

    def to_dict(self):
        return {
            rft_demo_file_name: self.demo_file,
            rft_round_number_name: self.round_number,
            rft_ct_bot_name: self.ct_bot,
            rft_one_bot_feature_store_index_name: self.one_bot_feature_store_index,
            rft_start_index_in_hdf5_name: self.start_index_in_hdf5,
            rft_length_name: self.length,
            rft_hdf5_key: self.hdf5_key,
            round_id_column: self.round_id
        }


rounds_for_traces: List[RoundForTrace] = [
    RoundForTrace("2351922_128038_bravos-vs-sharks-m2-dust2_12399afe-2607-11ec-84dc-0a58a9feac02.dem", 5, False, 1),
    RoundForTrace("2354429_133822_nexus-vs-ktrl-m1-dust2_7c7c8440-8dc1-11ec-872e-0a58a9feac02.dem", 30, False, 4),
    RoundForTrace("2349618_122963_100pg-vs-spirit-m3-dust2_92202a56-da9b-11eb-a928-0a58a9feac02.dem", 19, False, 2),
    RoundForTrace("2354576_136413_natus-vincere-vs-evil-geniuses-m2-dust2_7e07ebd2-b1e4-11ec-a36e-0a58a9feac02.dem", 23, False, 3),
    RoundForTrace("2358772_143226_ungentium-vs-anonymo-m2-dust2_57761380-3390-11ed-8f4d-0a58a9feac02.dem", 5, True, 3),
    RoundForTrace("2349702_123308_faze-vs-complexity-m1-dust2_64730b48-df4e-11eb-8b4e-0a58a9feac02.dem", 28, True, 1),
    RoundForTrace("2355240_135682_ihc-vs-the-shine-dust2_88e35aa0-aa0a-11ec-9c61-0a58a9feac02.dem", 9, True, 4),
    RoundForTrace("2348918_120880_1win-vs-izako-boars-m1-dust2_8b824c6c-be4c-11eb-86da-0a58a9feac02.dem", 9, True, 0),
]

trace_file_name = 'traces.hdf5'

def create_traces(loaded_model: LoadedModel):
    cur_start_index = 0
    result_dfs = []

    for i, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
        print(f"Processing hdf5 {i + 1} / {len(loaded_model.dataset.data_hdf5s)}: {hdf5_wrapper.hdf5_path}")
        loaded_model.cur_hdf5_index = i

        cur_hdf5_relevant = False
        cur_demo_names: List[str] = list(loaded_model.load_cur_hdf5_demo_names())
        for round_for_trace in rounds_for_traces:
            if round_for_trace.demo_file in cur_demo_names:
                cur_hdf5_relevant = True
                break

        if cur_hdf5_relevant:
            loaded_model.load_cur_hdf5_as_pd(load_cur_dataset=False, cast_bool_to_int=False)
            for round_for_trace in rounds_for_traces:
                if round_for_trace.demo_file in cur_demo_names:
                    game_id = cur_demo_names.index(round_for_trace.demo_file)
                    round_ticks_df = loaded_model.cur_loaded_df[
                        (loaded_model.cur_loaded_df[game_id_column] == game_id) &
                        (loaded_model.cur_loaded_df[round_number_column] == round_for_trace.round_number)]
                    result_dfs.append(round_ticks_df)
                    round_for_trace.start_index_in_hdf5 = cur_start_index
                    round_for_trace.length = len(round_ticks_df)
                    round_for_trace.hdf5_key = str(absolute_to_relative_train_test_key(hdf5_wrapper.hdf5_path))
                    round_for_trace.round_id = round_ticks_df[round_id_column].iloc[0]
                    cur_start_index += len(round_ticks_df)

    # make sure all rounds for traces have been found
    for i, round_for_trace in enumerate(rounds_for_traces):
        if round_for_trace.start_index_in_hdf5 == -1:
            print(f"invalid round_for_trace {i}")
    for round_for_trace in rounds_for_traces:
        assert round_for_trace.start_index_in_hdf5 != -1

    combined_df = pd.concat(result_dfs)
    extra_df = pd.DataFrame.from_records([rft.to_dict() for rft in rounds_for_traces])

    trace_path = load_data_result.multi_hdf5_wrapper.train_test_split_path.parent / trace_file_name
    save_pd_to_hdf5(trace_path, combined_df, extra_df=extra_df)


if __name__ == "__main__":
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, use_test_data_only=True)
    create_traces(loaded_model)
