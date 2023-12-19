from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from learn_bot.latent.engagement.column_names import game_id_column, round_id_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.vis.run_vis_checkpoint import load_data_options

all_statistics_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs' / 'all_statistics.csv'
all_players_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs' / 'all_players.csv'


def ticks_to_hours(ticks: int, tick_rate: int) -> float:
    return ticks / tick_rate / 60. / 60.


def compute_raw_statistics():
    all_statistics_df = pd.read_csv(all_statistics_data_path)
    all_statistics_sums_df = all_statistics_df.sum()

    num_ticks = all_statistics_sums_df["num ticks"]
    num_hours = ticks_to_hours(num_ticks, 128)
    num_shots = all_statistics_sums_df["num shots"]
    num_kills = all_statistics_sums_df["num kills"]

    all_players_df = pd.read_csv(all_players_data_path)
    all_players_set = all_players_df['player name'].nunique()

    print(f'num players {all_players_set}, num games {all_statistics_sums_df["num games"]}, num rounds {all_statistics_sums_df["num rounds"]}, '
          f'num ticks {num_ticks}, num hours {num_hours}, num shots {num_shots}, num kills {num_kills}')


def compute_plant_statistics():
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result)

    players_list: List[str] = []
    num_games = 0
    num_rounds = 0
    num_ticks = 0
    num_shots = 0
    num_kills = 0
    num_killeds = 0
    with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
        for i in range(len(loaded_model.dataset.data_hdf5s)):
            loaded_model.cur_hdf5_index = i
            loaded_model.load_cur_dataset_only()

            player_names_cur_dataset = loaded_model.load_cur_hdf5_player_names()

            num_games += loaded_model.get_cur_id_df()[game_id_column].nunique()
            num_rounds += loaded_model.get_cur_id_df()[round_id_column].nunique()
            num_ticks += len(loaded_model.get_cur_id_df())

            vis_df = loaded_model.get_cur_vis_df()
            player_ids: List[int] = []
            for player_index, player_player_area_columns in enumerate(specific_player_place_area_columns):
                all_player_id = vis_df[player_player_area_columns.player_id].astype('int32')
                valid_player_id = all_player_id[all_player_id >= 0]
                player_ids += valid_player_id.unique().tolist()
                num_shots += int(vis_df[player_player_area_columns.player_shots_cur_tick].sum())
                num_kills += int(vis_df[player_player_area_columns.player_kill_next_tick].sum())
                num_killeds += int(vis_df[player_player_area_columns.player_killed_next_tick].sum())
                player_alive = loaded_model.cur_dataset.X[:, loaded_model.model.alive_columns[player_index]]
                num_killeds_when_already_dead = \
                    sum((vis_df[player_player_area_columns.player_killed_next_tick] > 0.5) & (player_alive < 0.5))
                if num_killeds_when_already_dead > 0:
                    print('killed when already dead')

            unique_player_ids = set(player_ids)
            # first name is invalid, so take names after that, just how I structured names array
            players_list += [player_names_cur_dataset[i+1] for i in unique_player_ids]

            pbar.update(1)

    num_hours = ticks_to_hours(num_ticks, 16)

    players_set = set(players_list)

    print(f'num players {len(players_set)}, num games {num_games}, num rounds {num_rounds}, '
          f'num ticks {num_ticks}, num hours {num_hours}, num shots {num_shots}, num kills {num_kills}, num killeds {num_killeds}')


if __name__ == "__main__":
    compute_raw_statistics()
    compute_plant_statistics()