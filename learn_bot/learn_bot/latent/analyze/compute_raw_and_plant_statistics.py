from math import ceil
from pathlib import Path
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import plot_hist, generate_bins
from learn_bot.latent.analyze.run_coverage import get_round_starts_np
from learn_bot.latent.engagement.column_names import game_id_column, round_id_column, tick_id_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.vis.run_vis_checkpoint import load_data_options
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

all_statistics_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs' / 'all_statistics.csv'
all_players_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs' / 'all_players.csv'


def ticks_to_hours(ticks: int, tick_rate: int) -> float:
    return ticks / tick_rate / 60. / 60.


def ticks_to_seconds(ticks: int, tick_rate: int) -> float:
    return ticks / tick_rate


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
    num_ticks_per_round = []
    num_ct_players_per_round = []
    num_t_players_per_round = []
    with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
        for i in range(len(loaded_model.dataset.data_hdf5s)):
            loaded_model.cur_hdf5_index = i
            loaded_model.load_cur_dataset_only()

            player_names_cur_dataset = loaded_model.load_cur_hdf5_player_names()

            num_games += loaded_model.get_cur_id_df()[game_id_column].nunique()
            num_rounds += loaded_model.get_cur_id_df()[round_id_column].nunique()
            num_ticks += len(loaded_model.get_cur_id_df())

            # compute key event metrics
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
                    print(f'killed when already dead {num_killeds_when_already_dead}')
                num_kills_when_already_dead = \
                    sum((vis_df[player_player_area_columns.player_kill_next_tick] > 0.5) & (player_alive < 0.5))
                if num_kills_when_already_dead > 0:
                    print(f'kill when already dead {num_kills_when_already_dead}')

            unique_player_ids = set(player_ids)
            # first name is invalid, so take names after that, just how I structured names array
            players_list += [player_names_cur_dataset[i+1] for i in unique_player_ids]

            # compute per round metrics
            num_ticks_per_round += list(loaded_model.get_cur_id_df().groupby(round_id_column)[tick_id_column].count())
            X_np = get_round_starts_np(loaded_model)
            ct_alive_np = X_np[:, loaded_model.model.alive_columns[:loaded_model.model.num_players_per_team]]
            num_ct_players_per_round += ct_alive_np.sum(axis=1).tolist()
            t_alive_np = X_np[:, loaded_model.model.alive_columns[loaded_model.model.num_players_per_team:]]
            num_t_players_per_round += t_alive_np.sum(axis=1).tolist()

            pbar.update(1)

    num_hours = ticks_to_hours(num_ticks, 16)

    players_set = set(players_list)

    players_per_round_df = pd.DataFrame.from_dict({'ct per round': num_ct_players_per_round,
                                                   't per round': num_t_players_per_round})
    players_per_round_df['players per round'] = players_per_round_df['ct per round'] + \
                                                players_per_round_df['t per round']

    # timing mismatch can add an extra second, cap that off
    num_seconds_per_round = [min(40., ticks_to_seconds(nt, 16)) for nt in num_ticks_per_round]
    seconds_per_round_series = pd.Series(num_seconds_per_round)

    # plot histograms of players per round and round lengths
    # players per round
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    fig = plt.figure(figsize=(3.25, 6/8 * 3.25 * 0.49), constrained_layout=True)
    axs = fig.subplots(1, 3, squeeze=False, sharey=True)

    #fig.suptitle('Round Statistics', fontsize=8)

    # plot ct per round
    plot_hist(axs[0, 0], players_per_round_df['ct per round'], [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    axs[0, 0].set_title('Initial Offense', fontsize=8, x=0.45)
    axs[0, 0].set_xlabel('Players', fontsize=8, labelpad=2)
    axs[0, 0].set_ylabel('Percent Of Rounds', fontsize=7, labelpad=2)
    #axs[0, 0].text(3, 0.2, players_per_round_df['ct per round'].describe().to_string())
    axs[0, 0].set_ylim(0, 0.4)
    axs[0, 0].set_xlim(0.5, 5.5)

    axs[0, 0].set_xticks([1, 3, 5])
    axs[0, 0].set_yticks([0, 0.2, 0.4])
    axs[0, 0].tick_params(axis="x", labelsize=8, pad=1)
    axs[0, 0].tick_params(axis="y", labelsize=8, pad=1)

    # remove right/top spine
    axs[0, 0].spines['top'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    axs[0, 0].yaxis.grid(False)#True, color='#EEEEEE', dashes=[4, 1])
    axs[0, 0].xaxis.grid(False)

    # plot t per round
    plot_hist(axs[0, 1], players_per_round_df['t per round'], [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    axs[0, 1].set_title('Initial Defense', fontsize=8, x=0.55)
    axs[0, 1].set_xlabel('Players', fontsize=8, labelpad=2)
    #axs[0, 1].set_ylabel('Percent')
    #axs[0, 1].text(3, 0.2, players_per_round_df['t per round'].describe().to_string())
    axs[0, 1].set_ylim(0, 0.4)
    axs[0, 1].set_xlim(0.5, 5.5)

    axs[0, 1].set_xticks([1, 3, 5])
    axs[0, 1].tick_params(axis="x", labelsize=8, pad=1)

    # remove right/top spine
    axs[0, 1].spines['top'].set_visible(False)
    axs[0, 1].spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    axs[0, 1].yaxis.grid(False)
    axs[0, 1].xaxis.grid(False)

    # round lengths

    seconds_per_round_bins = generate_bins(0, int(ceil(max(seconds_per_round_series))), 5)
    plot_hist(axs[0, 2], seconds_per_round_series, seconds_per_round_bins)
    axs[0, 2].set_title('Length', fontsize=8)
    axs[0, 2].set_xlabel('Seconds', fontsize=8, labelpad=2)
    #axs[0, 2].set_ylabel('Percent')
    #axs[2, 0].text(3, 0.2, seconds_per_round_series.describe().to_string())
    axs[0, 2].set_xlim(0, 40)
    axs[0, 2].set_ylim(0, 0.4)

    axs[0, 2].set_xticks([0, 20, 40])
    axs[0, 2].tick_params(axis="x", labelsize=8, pad=1)

    # remove right/top spine
    axs[0, 2].spines['top'].set_visible(False)
    axs[0, 2].spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    axs[0, 2].yaxis.grid(False)
    axs[0, 2].xaxis.grid(False)

    plt.savefig(Path(__file__).parent / 'plots' / 'plant_statistics.pdf')
    plt.close(fig)

    print(f'num players {len(players_set)}, num games {num_games}, num rounds {num_rounds}, '
          f'num ticks {num_ticks}, num hours {num_hours}, num shots {num_shots}, '
          f'num kills {num_kills}, num killeds {num_killeds}, '
          f'mean players per round {players_per_round_df["players per round"].mean()}, '
          f'mean round length {seconds_per_round_series.mean()} (s)')


if __name__ == "__main__":
    compute_raw_statistics()
    compute_plant_statistics()