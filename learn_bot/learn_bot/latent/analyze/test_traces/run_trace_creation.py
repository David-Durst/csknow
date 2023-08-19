from learn_bot.latent.engagement.column_names import game_id_column, round_number_column
from learn_bot.latent.place_area.simulator import *
from dataclasses import dataclass


@dataclass
class RoundForTrace:
    demo_file: str
    round_number: int


rounds_for_traces: List[RoundForTrace] = [
    RoundForTrace("2354544_135491_vitality-vs-ence-m2-dust2_d20d7c6e-a6e6-11ec-8233-0a58a9feac02.dem", 25),
    RoundForTrace("2355669_136889_masonic-vs-young-ninjas-m2-dust2_9364d3e6-bb53-11ec-a7e7-0a58a9feac02.dem", 6)
]

def create_traces(loaded_model: LoadedModel):
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
            loaded_model.load_cur_hdf5_as_pd(load_cur_dataset=False)
            for round_for_trace in rounds_for_traces:
                if round_for_trace.demo_file in cur_demo_names:
                    game_id = cur_demo_names.index(round_for_trace.demo_file)
                    round_ticks_df = loaded_model.cur_loaded_df[
                        (loaded_model.cur_loaded_df[game_id_column] == game_id) &
                        (loaded_model.cur_loaded_df[round_number_column] == round_for_trace.round_number)]
                    for player_columns in specific_player_place_area_columns:
                        




if __name__ == "__main__":
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, use_test_data_only=True)
    create_traces(loaded_model)
