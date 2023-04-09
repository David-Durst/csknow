from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from learn_bot.latent.dataset import latent_hdf5_data_path
from learn_bot.latent.engagement.column_names import get_ith_enemy_columns
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.latent.train import plot_path

rounds_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'rounds.hdf5'
weapon_fire_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'weaponFire.hdf5'
kills_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'kills.hdf5'
hurt_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'hurt.hdf5'
latent_engagement_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'latentEngagement.hdf5'
inference_latent_engagement_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'inferenceLatentEngagement.hdf5'

valid_round_numbers = [2]

def match_lims(ax0: plt.Axes, ax1: plt.Axes):
    xs0 = ax0.get_xlim()
    ys0 = ax0.get_ylim()
    xs1 = ax1.get_xlim()
    ys1 = ax1.get_ylim()
    ax0.set_xlim(min(xs0[0], xs1[0]), max(xs0[1], xs1[1]))
    ax1.set_xlim(min(xs0[0], xs1[0]), max(xs0[1], xs1[1]))
    ax0.set_ylim(min(ys0[0], ys1[0]), max(ys0[1], ys1[1]))
    ax1.set_ylim(min(ys0[0], ys1[0]), max(ys0[1], ys1[1]))


def compare_results(ticks_df: pd.DataFrame, rounds_df: pd.DataFrame, weapon_fire_df: pd.DataFrame, kills_df: pd.DataFrame, hurt_df: pd.DataFrame,
                    latent_engagement_df: pd.DataFrame, inference_latent_engagement_df: pd.DataFrame):
    # percent of ticks data with target
    with_no_enemy_ticks_df = ticks_df[
        (ticks_df[get_ith_enemy_columns(0).engagement_state] == 3) &
        (ticks_df[get_ith_enemy_columns(1).engagement_state] == 3) &
        (ticks_df[get_ith_enemy_columns(2).engagement_state] == 3) &
        (ticks_df[get_ith_enemy_columns(3).engagement_state] == 3) &
        (ticks_df[get_ith_enemy_columns(4).engagement_state] == 3)
    ]
    print(f'''percent no enemy: {len(with_no_enemy_ticks_df)/len(ticks_df)}''')
    # get data in valid rounds
    valid_rounds_df = rounds_df[(rounds_df['game id'] == 0) & rounds_df['round number'].isin(valid_round_numbers)]
    round_start_ticks = valid_rounds_df.loc[:, 'start tick'].tolist()
    round_end_ticks = valid_rounds_df.loc[:, 'end tick'].tolist()
    round_ranges = zip(round_start_ticks, round_end_ticks)
    valid_latent_engagement = latent_engagement_df['start tick id'] < -1
    valid_inference_latent_engagement = inference_latent_engagement_df['start tick id'] < -1
    valid_kills = kills_df['tick id'] < -1
    valid_hurt = hurt_df['tick id'] < -1
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
        valid_hurt = valid_hurt | (
            hurt_df['tick id'].between(round_range[0], round_range[1])
        )

    valid_latent_engagement_df = latent_engagement_df[valid_latent_engagement].copy()
    valid_inference_latent_engagement_df = inference_latent_engagement_df[valid_inference_latent_engagement]
    valid_kills_df = kills_df[valid_kills].copy()
    valid_hurt_df = hurt_df[valid_hurt].copy()

    # graph engagement length
    valid_latent_engagement_df['Engagement Length'] = \
        valid_latent_engagement_df['tick length'] / 128.
    bins=[i*0.5 for i in range(2*7)]
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(2*8, 5 * 8))
    valid_latent_engagement_df.hist(column='Engagement Length', bins=bins, ax=ax[0,0])
    ax[0,0].set_xlabel("seconds")
    ax[0,0].set_ylabel("number of engagements")
    ax[0,0].set_title("Latent Engagements Length")
    ax[0,0].text(3., 40., valid_latent_engagement_df['Engagement Length'].describe().to_string())
    #ax[0,0].figure.savefig(plot_path / 'latent_engagement_length.png')
    valid_inference_latent_engagement_df['Engagement Length'] = \
        valid_inference_latent_engagement_df['tick length'] / 128.
    valid_inference_latent_engagement_df.hist(column='Engagement Length', bins=bins, ax=ax[0,1])
    ax[0,1].set_xlabel("seconds")
    ax[0,1].set_ylabel("number of engagements")
    ax[0,1].set_title("Inference Latent Engagements Length")
    ax[0,1].text(3., 40., valid_inference_latent_engagement_df['Engagement Length'].describe().to_string())
    match_lims(ax[0,0], ax[0,1])

    # graph percent match nearest crosshair enemy cur tick
    nearest_crosshair_enemy_col = "percent match nearest crosshair enemy cur tick"
    bins=[i / 20. for i in range(20)]
    valid_latent_engagement_df.hist(column=nearest_crosshair_enemy_col, bins=bins, ax=ax[1, 0])
    ax[1,0].set_xlabel("percent of frames")
    ax[1,0].set_ylabel("number of engagements")
    ax[1,0].set_title("Percent of Frames Match Nearest Crosshair Enemy Cur Tick")
    ax[1,0].text(0.3, 0.5, valid_latent_engagement_df[nearest_crosshair_enemy_col].describe().to_string(),
                 transform=ax[1,0].transAxes)
    valid_inference_latent_engagement_df.hist(column=nearest_crosshair_enemy_col, bins=bins, ax=ax[1,1])
    ax[1,1].set_xlabel("percent of frames")
    ax[1,1].set_ylabel("number of engagements")
    ax[1,1].set_title("Inference Percent of Frames Match Nearest Crosshair Enemy Cur Tick")
    ax[1,1].text(0.3, 0.5, valid_inference_latent_engagement_df[nearest_crosshair_enemy_col].describe().to_string(),
                 transform=ax[1,1].transAxes)
    match_lims(ax[1,0], ax[1,1])
    print(valid_inference_latent_engagement_df[valid_inference_latent_engagement_df[nearest_crosshair_enemy_col] < 0.1].to_string())

    # graph percent match nearest crosshair enemy 500ms
    nearest_crosshair_enemy_col = "percent match nearest crosshair enemy 500ms"
    bins=[i / 20. for i in range(20)]
    valid_latent_engagement_df.hist(column=nearest_crosshair_enemy_col, bins=bins, ax=ax[2,0])
    ax[2,0].set_xlabel("percent of frames")
    ax[2,0].set_ylabel("number of engagements")
    ax[2,0].set_title("Percent of Frames Match Nearest Crosshair Enemy 500ms")
    ax[2,0].text(0.3, 0.5, valid_latent_engagement_df[nearest_crosshair_enemy_col].describe().to_string(),
                 transform=ax[2,0].transAxes)
    valid_inference_latent_engagement_df.hist(column=nearest_crosshair_enemy_col, bins=bins, ax=ax[2,1])
    ax[2,1].set_xlabel("percent of frames")
    ax[2,1].set_ylabel("number of engagements")
    ax[2,1].set_title("Inference Percent of Frames Match Nearest Crosshair Enemy 500ms")
    ax[2,1].text(0.3, 0.5, valid_inference_latent_engagement_df[nearest_crosshair_enemy_col].describe().to_string(),
                 transform=ax[2,1].transAxes)
    match_lims(ax[2,0], ax[2,1])

    # graph percent match nearest crosshair enemy 1s
    nearest_crosshair_enemy_col = "percent match nearest crosshair enemy 1s"
    bins=[i / 20. for i in range(20)]
    valid_latent_engagement_df.hist(column=nearest_crosshair_enemy_col, bins=bins, ax=ax[3,0])
    ax[3,0].set_xlabel("percent of frames")
    ax[3,0].set_ylabel("number of engagements")
    ax[3,0].set_title("Percent of Frames Match Nearest Crosshair Enemy 1s")
    ax[3,0].text(0.3, 0.5, valid_latent_engagement_df[nearest_crosshair_enemy_col].describe().to_string(),
                 transform=ax[3,0].transAxes)
    valid_inference_latent_engagement_df.hist(column=nearest_crosshair_enemy_col, bins=bins, ax=ax[3,1])
    ax[3,1].set_xlabel("percent of frames")
    ax[3,1].set_ylabel("number of engagements")
    ax[3,1].set_title("Inference Percent of Frames Match Nearest Crosshair Enemy 1s")
    ax[3,1].text(0.3, 0.5, valid_inference_latent_engagement_df[nearest_crosshair_enemy_col].describe().to_string(),
                 transform=ax[3,1].transAxes)
    match_lims(ax[3,0], ax[3,1])

    # graph percent match nearest crosshair enemy 2s
    nearest_crosshair_enemy_col = "percent match nearest crosshair enemy 2s"
    bins=[i / 20. for i in range(20)]
    valid_latent_engagement_df.hist(column=nearest_crosshair_enemy_col, bins=bins, ax=ax[4,0])
    ax[4,0].set_xlabel("percent of frames")
    ax[4,0].set_ylabel("number of engagements")
    ax[4,0].set_title("Percent of Frames Match Nearest Crosshair Enemy 2s")
    describe_str = valid_latent_engagement_df[nearest_crosshair_enemy_col].describe().to_string()
    ticks_per_engagement_2s_correct = \
        valid_latent_engagement_df[nearest_crosshair_enemy_col] * valid_latent_engagement_df['tick length']
    percent_ticks_2s_correct = ticks_per_engagement_2s_correct.sum() / valid_latent_engagement_df['tick length'].sum()
    frame_weight_str = '''pct frames nearest enemy 2s:''' + \
                       f'''{percent_ticks_2s_correct:.4f}'''
    ax[4,0].text(0.3, 0.5, describe_str + "\n" + frame_weight_str, transform=ax[4,0].transAxes)
    valid_inference_latent_engagement_df.hist(column=nearest_crosshair_enemy_col, bins=bins, ax=ax[4,1])
    ax[4,1].set_xlabel("percent of frames")
    ax[4,1].set_ylabel("number of engagements")
    ax[4,1].set_title("Inference Percent of Frames Match Nearest Crosshair Enemy 2s")
    describe_str = valid_inference_latent_engagement_df[nearest_crosshair_enemy_col].describe().to_string()
    ticks_per_inference_engagement_2s_correct = \
        valid_inference_latent_engagement_df[nearest_crosshair_enemy_col] * valid_inference_latent_engagement_df['tick length']
    inference_percent_ticks_2s_correct = ticks_per_inference_engagement_2s_correct.sum() / valid_inference_latent_engagement_df['tick length'].sum()
    frame_weight_str = '''pct frames nearest enemy 2s:''' + \
                       f'''{inference_percent_ticks_2s_correct:.4f}'''
    ax[4,1].text(0.3, 0.5, describe_str + "\n" + frame_weight_str, transform=ax[4,1].transAxes)
    match_lims(ax[4,0], ax[4,1])
    plt.tight_layout()
    fig.savefig(plot_path / 'inference_vs_heuristic_engagements.png')

    # measure kill agreement
    num_engagement_kill_matches = 0
    missing_engagement_kills = []
    num_inference_engagement_kill_matches = 0
    missing_inference_engagement_kills = []
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
            missing_engagement_kills.append(i)
        #    print("latent engagement missing")

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
            missing_inference_engagement_kills.append(i)
        #    print(f'''inference latent engagement missing {valid_kills_df.iloc[i].to_string()}''')
    print(f'''kills {len(valid_kills_df)}, latent engagement matches {num_engagement_kill_matches}, inference latent engagement matches {num_inference_engagement_kill_matches}''')
    if len(missing_engagement_kills) > 0:
        print("latent missing")
        print(valid_hurt_df.iloc[missing_engagement_kills].to_string())
    if len(missing_inference_engagement_kills) > 0:
        print("inference latent missing")
        print(valid_hurt_df.iloc[missing_inference_engagement_kills].to_string())

    # measure hurt agreement
    num_engagement_hurt_matches = 0
    missing_engagement_hurt = []
    num_inference_engagement_hurt_matches = 0
    missing_inference_engagement_hurt = []
    for i in range(len(valid_hurt_df)):
        attacker = valid_hurt_df.iloc[i].loc['attacker']
        victim = valid_hurt_df.iloc[i].loc['victim']
        hurt_tick_id = valid_hurt_df.iloc[i].loc['tick id']
        latent_engagement_row = valid_latent_engagement_df[
            (valid_latent_engagement_df['attacker id'] == attacker) &
            (valid_latent_engagement_df['target id'] == victim) &
            (valid_latent_engagement_df['start tick id'] <= hurt_tick_id) &
            (valid_latent_engagement_df['end tick id'] >= hurt_tick_id - 1)]
        if len(latent_engagement_row) > 1:
            print("latent engagement not unique")
        elif len(latent_engagement_row) == 1:
            num_engagement_hurt_matches += 1
        else:
            missing_engagement_hurt.append(i)
        #    print("latent engagement missing")

        inference_latent_engagement_row = valid_inference_latent_engagement_df[
            (valid_inference_latent_engagement_df['attacker id'] == attacker) &
            (valid_inference_latent_engagement_df['target id'] == victim) &
            (valid_inference_latent_engagement_df['start tick id'] <= hurt_tick_id) &
            (valid_inference_latent_engagement_df['end tick id'] >= hurt_tick_id - 1)]

        if len(inference_latent_engagement_row) > 1:
            print("inference latent engagement not unique")
        elif len(inference_latent_engagement_row) == 1:
            num_inference_engagement_hurt_matches += 1
        else:
            missing_inference_engagement_hurt.append(i)
        #    print(f'''inference latent engagement missing {valid_kills_df.iloc[i].to_string()}''')
    print(f'''hurt {len(valid_hurt_df)}, latent engagement matches {num_engagement_hurt_matches}, inference latent engagement matches {num_inference_engagement_hurt_matches}''')
    if len(missing_engagement_hurt) > 0:
        print("latent missing")
        print(valid_hurt_df.iloc[missing_engagement_hurt].to_string())
    if len(missing_inference_engagement_hurt) > 0:
        print("inference latent missing")
        print(valid_hurt_df.iloc[missing_inference_engagement_hurt].to_string())


if __name__ == "__main__":
    ticks_df = load_hdf5_to_pd(latent_hdf5_data_path)
    ticks_df = ticks_df[ticks_df['valid'] == 1.]
    rounds_df = load_hdf5_to_pd(rounds_hdf5_data_path)
    weapon_fire_df = load_hdf5_to_pd(weapon_fire_hdf5_data_path)
    kills_df = load_hdf5_to_pd(kills_hdf5_data_path)
    hurt_df = load_hdf5_to_pd(hurt_hdf5_data_path)
    latent_engagement_df = load_hdf5_to_pd(latent_engagement_hdf5_data_path)
    inference_latent_engagement_df = load_hdf5_to_pd(inference_latent_engagement_hdf5_data_path)
    compare_results(ticks_df, rounds_df, weapon_fire_df, kills_df, hurt_df, latent_engagement_df, inference_latent_engagement_df)