from pathlib import Path

import pandas as pd

from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.latent.train import plot_path

rounds_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'rounds.hdf5'
weapon_fire_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'weaponFire.hdf5'
kills_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'kills.hdf5'
latent_engagement_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'latentEngagement.hdf5'
inference_latent_engagement_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'inferenceLatentEngagement.hdf5'

valid_round_numbers = [2]


def compare_results(rounds_df: pd.DataFrame, weapon_fire_df: pd.DataFrame, kills_df: pd.DataFrame,
                    latent_engagement_df: pd.DataFrame, inference_latent_engagement_df: pd.DataFrame):
    # get data in valid rounds
    valid_rounds_df = rounds_df[(rounds_df['game id'] == 0) & rounds_df['round number'].isin(valid_round_numbers)]
    round_start_ticks = valid_rounds_df.loc[:, 'start tick'].tolist()
    round_end_ticks = valid_rounds_df.loc[:, 'end tick'].tolist()
    round_ranges = zip(round_start_ticks, round_end_ticks)
    valid_latent_engagement = latent_engagement_df['start tick id'] < -1
    valid_inference_latent_engagement = inference_latent_engagement_df['start tick id'] < -1
    valid_kills = kills_df['tick id'] < -1
    for round_range in round_ranges:
        valid_latent_engagement = valid_latent_engagement | (
                latent_engagement_df['start tick id'].between(round_range[0], round_range[1])
        )
        valid_inference_latent_engagement = valid_inference_latent_engagement | (
            inference_latent_engagement_df['start tick id'].between(round_range[0], round_range[1])
        )
        valid_kills = valid_kills | (
            kills_df['tick id'].between(round_range[0], round_range[1])
        )

    valid_latent_engagement_df = latent_engagement_df[valid_latent_engagement].copy()
    valid_inference_latent_engagement_df = inference_latent_engagement_df[valid_inference_latent_engagement]
    valid_kills_df = kills_df[valid_kills].copy()

    # graph engagement length
    valid_latent_engagement_df['Engagement Length'] = \
        valid_latent_engagement_df['tick length'] / 128.
    bins=[i*0.5 for i in range(2*7)]
    ax = valid_latent_engagement_df.hist(column='Engagement Length', bins=bins)
    ax[0,0].set_xlabel("seconds")
    ax[0,0].set_ylabel("number of engagements")
    ax[0,0].set_title("Latent Engagements Length")
    ax[0,0].text(3., 40., str(valid_latent_engagement_df['Engagement Length'].describe()))
    ax[0,0].figure.savefig(plot_path / 'latent_engagement_length.png')
    valid_inference_latent_engagement_df['Engagement Length'] = \
        valid_inference_latent_engagement_df['tick length'] / 128.
    ax = valid_inference_latent_engagement_df.hist(column='Engagement Length', bins=bins)
    ax[0,0].set_xlabel("seconds")
    ax[0,0].set_ylabel("number of engagements")
    ax[0,0].set_title("Inference Latent Engagements Length")
    ax[0,0].text(3., 40., str(valid_inference_latent_engagement_df['Engagement Length'].describe()))
    ax[0,0].figure.savefig(plot_path / 'inference_latent_engagement_length.png')

    # measure kill agreement
    num_engagement_kill_matches = 0
    num_inference_engagement_kill_matches = 0
    for i in range(len(valid_kills_df)):
        killer = valid_kills_df.iloc[i].loc['killer']
        victim = valid_kills_df.iloc[i].loc['victim']
        kill_tick_id = valid_kills_df.iloc[i].loc['tick id']
        latent_engagement_row = valid_latent_engagement_df[
            (valid_latent_engagement_df['attacker id'] == killer) &
            (valid_latent_engagement_df['target id'] == victim) &
            (valid_latent_engagement_df['start tick id'] <= kill_tick_id) &
            (valid_latent_engagement_df['end tick id'] >= kill_tick_id - 1)]
        if len(latent_engagement_row) > 1:
            print("latent engagement not unique")
        elif len(latent_engagement_row) == 1:
            num_engagement_kill_matches += 1
        else:
            print("latent engagement missing")

        inference_latent_engagement_row = valid_inference_latent_engagement_df[
            (valid_inference_latent_engagement_df['attacker id'] == killer) &
            (valid_inference_latent_engagement_df['target id'] == victim) &
            (valid_inference_latent_engagement_df['start tick id'] <= kill_tick_id) &
            (valid_inference_latent_engagement_df['end tick id'] >= kill_tick_id - 1)]

        if len(inference_latent_engagement_row) > 1:
            print("inference latent engagement not unique")
        elif len(inference_latent_engagement_row) == 1:
            num_inference_engagement_kill_matches += 1
        else:
            print(f'''inference latent engagement missing {valid_kills_df.iloc[i].to_string()}''')
    print(f'''kills {len(valid_kills_df)}, latent engagement matches {num_engagement_kill_matches}, inference latent engagement matches {num_inference_engagement_kill_matches}''')





if __name__ == "__main__":
    rounds_df = load_hdf5_to_pd(rounds_hdf5_data_path)
    weapon_fire_df = load_hdf5_to_pd(weapon_fire_hdf5_data_path)
    kills_df = load_hdf5_to_pd(kills_hdf5_data_path)
    latent_engagement_df = load_hdf5_to_pd(latent_engagement_hdf5_data_path)
    inference_latent_engagement_df = load_hdf5_to_pd(inference_latent_engagement_hdf5_data_path)
    compare_results(rounds_df, weapon_fire_df, kills_df, latent_engagement_df, inference_latent_engagement_df)
