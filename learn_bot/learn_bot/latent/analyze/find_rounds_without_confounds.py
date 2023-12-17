from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import get_hdf5_to_test_round_ids
from learn_bot.latent.engagement.column_names import tick_id_column, round_id_column, game_id_column, \
    round_number_column, game_tick_number_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.column_names import grenade_columns, grenade_throw_tick_col, grenade_active_tick_col, \
    grenade_expired_tick_col, grenade_destroy_tick_col
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.vis.run_vis_checkpoint import load_data_options
from learn_bot.libs.pd_printing import set_pd_print_options

demo_file_column = 'demo file'
num_ticks_column = 'num ticks'

def find_rounds_without_confounds():
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result, use_test_data_only=True)

    set_pd_print_options()
    pd.set_option('display.max_colwidth', None)
    hdf5_to_round_ids = get_hdf5_to_test_round_ids(push_only=True)[0]
    push_no_grenade_round_dicts: List[Dict] = []

    with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
        for i in range(len(loaded_model.dataset.data_hdf5s)):
            if i > 0:
                break
            loaded_model.cur_hdf5_index = i
            loaded_model.load_cur_dataset_only()

            id_df = loaded_model.get_cur_id_df()

            push_round_ids = hdf5_to_round_ids[str(loaded_model.dataset.data_hdf5s[i].hdf5_path.name)]

            grenades_df = loaded_model.get_cur_extra_df(grenade_columns)
            throw_tick_ids = grenades_df[grenade_throw_tick_col].unique()
            active_tick_ids = grenades_df[grenade_active_tick_col].unique()
            expired_tick_ids = grenades_df[grenade_expired_tick_col].unique()
            destroy_tick_ids = grenades_df[grenade_destroy_tick_col].unique()

            grenade_id_df = id_df[(id_df[tick_id_column].isin(throw_tick_ids)) |
                                  (id_df[tick_id_column].isin(active_tick_ids)) |
                                  (id_df[tick_id_column].isin(expired_tick_ids)) |
                                  (id_df[tick_id_column].isin(destroy_tick_ids))]
            grenade_rounds = grenade_id_df[round_id_column].unique()
            push_no_grenade_all_ticks_in_rounds_id_df = id_df[~id_df[round_id_column].isin(grenade_rounds) &
                                                              id_df[round_id_column].isin(push_round_ids)]
            push_no_grenade_round_summaries_df = push_no_grenade_all_ticks_in_rounds_id_df.groupby(round_id_column).agg({
                game_id_column: 'first',
                round_id_column: 'first',
                round_number_column: 'first',
                game_tick_number_column: 'first',
                tick_id_column: 'count'
            })

            for _, round_row in push_no_grenade_round_summaries_df.iterrows():
                push_no_grenade_round_dicts.append({
                    'hdf5_index': i,
                    demo_file_column: loaded_model.cur_demo_names[round_row[game_id_column]],
                    round_id_column: round_row[round_id_column],
                    round_number_column: round_row[round_number_column],
                    game_tick_number_column: round_row[game_tick_number_column],
                    num_ticks_column: round_row[tick_id_column]
                })

            # TODO: merge this with push rounds so I know where to get bot examples

            pbar.update(1)

    result_df = pd.DataFrame.from_records(push_no_grenade_round_dicts)
    #result_df = result_df.sample(frac=1, random_state=42).iloc[0:10]
    print(result_df)


if __name__ == "__main__":
    find_rounds_without_confounds()
