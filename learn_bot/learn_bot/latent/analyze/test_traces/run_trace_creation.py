from learn_bot.latent.engagement.column_names import game_id_column, round_number_column
from learn_bot.latent.order.column_names import c4_pos_cols
from learn_bot.latent.place_area.simulator import *
from dataclasses import dataclass

from learn_bot.libs.df_grouping import make_index_column
from learn_bot.libs.hdf5_to_pd import save_pd_to_hdf5


@dataclass
class RoundForTrace:
    demo_file: str
    round_number: int
    start_index_in_hdf5: int = -1
    length: int = -1

    def to_dict(self):
        return {
            "demo file": self.demo_file,
            "round number": self.round_number,
            "start index in hdf5": self.start_index_in_hdf5,
            "length": self.length
        }


rounds_for_traces: List[RoundForTrace] = [
    # _28, 0/52, round id 84 (i think, roughly)
    RoundForTrace("2354548_135549_outsiders-vs-ence-m2-dust2_9c77e360-a7ad-11ec-8656-0a58a9feac02.dem", 20),
    # _31, 1/52, round id 25
    RoundForTrace("2355685_136862_1620-kings-vs-dynasty-m2-dust2_1241692a-bb56-11ec-8b31-0a58a9feac02.dem", 0)
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
                    cur_start_index += len(round_ticks_df)

    # make sure all rounds for traces have been found
    for round_for_trace in rounds_for_traces:
        assert round_for_trace.start_index_in_hdf5 != -1

    combined_df = pd.concat(result_dfs)
    extra_df = pd.DataFrame.from_records([rft.to_dict() for rft in rounds_for_traces])

    trace_path = \
        load_data_result.multi_hdf5_wrapper.train_test_split_path.parent / trace_file_name
    save_pd_to_hdf5(trace_path, combined_df, extra_df=extra_df)


if __name__ == "__main__":
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, use_test_data_only=True)
    create_traces(loaded_model)
