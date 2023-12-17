from typing import Dict, List

from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import get_hdf5_to_test_round_ids
from learn_bot.latent.engagement.column_names import tick_id_column, round_id_column, game_id_column, \
    round_number_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.column_names import grenade_columns, grenade_throw_tick_col
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.vis.run_vis_checkpoint import load_data_options

demo_file_column = 'demo file'
num_ticks_column = 'num ticks'

def find_rounds_without_confounds():
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, use_test_data_only=True)

    hdf5_to_round_ids = get_hdf5_to_test_round_ids(push_only=True)[0]
    push_no_grenade_round_dicts: List[Dict] = []

    with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
        for i in range(len(loaded_model.dataset.data_hdf5s)):
            loaded_model.cur_hdf5_index = i
            loaded_model.load_cur_dataset_only()

            id_df = loaded_model.get_cur_id_df()

            push_round_ids = hdf5_to_round_ids[str(loaded_model.dataset.data_hdf5s[i].hdf5_path.name)]

            player_names_cur_dataset = loaded_model.load_cur_hdf5_player_names()
            grenades_df = loaded_model.get_cur_extra_df(grenade_columns)
            throw_tick_ids = grenades_df[grenade_throw_tick_col].unique()

            push_no_grenade_id_df = id_df[id_df[tick_id_column].isin(throw_tick_ids) &
                                          id_df[round_id_column].isin(push_round_ids)]
            push_no_grenade_rounds_df = push_no_grenade_id_df.groupby(round_id_column).agg({
                game_id_column: 'first',
                round_id_column: 'first',
                round_number_column: 'first',
                tick_id_column: 'count'
            })

            for i, round_row in push_no_grenade_rounds_df.iterrows():
                push_no_grenade_round_dicts.append({
                    demo_file_column: loaded_model.cur_demo_names[round_row[game_id_column]],
                    round_id_column: round_row[round_id_column],
                    round_number_column: round_row[round_number_column],
                    num_ticks_column: round_row[tick_id_column]
                })

            pbar.update(1)

    num_hours = ticks_to_hours(num_ticks, 16)

    players_set = set(players_list)

    print(f'num players {len(players_set)}, num games {num_games}, num rounds {num_rounds}, '
          f'num ticks {num_ticks}, num hours {num_hours}, num shots {num_shots}, num kills {num_kills}')


if __name__ == "__main__":
    find_rounds_without_confounds()
