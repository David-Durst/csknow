from pathlib import Path

import pandas as pd
import os
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

from learn_bot.engagement_aim.analysis.fitts import test_name_col
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.place_area.column_names import test_success_col, specific_player_place_area_columns
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.latent.place_area.load_data import manual_latent_team_hdf5_data_path, rollout_latent_team_hdf5_data_path
from learn_bot.mining.area_cluster import d2_radar_path, Vec3, MapCoordinate
import matplotlib as mpl
import matplotlib.pyplot as plt

plots_path = Path(__file__).parent / 'plots'


RoundsPerTestResult = Dict[str, Dict[bool, List[int]]]


def get_rounds_per_test_result(data_df: pd.DataFrame) -> RoundsPerTestResult:
    grouped_df = data_df.groupby([test_name_col, test_success_col])[round_id_column].agg('unique')
    group_items = list(grouped_df.items())
    result = {}
    for k, v in group_items:
        if k[0] not in result:
            result[k[0]] = {}
        success = k[1] > 0.5
        result[k[0]][success] = list(v)

    for test_name in result:
        num_success = 0
        num_failure = 0
        if True in result[test_name]:
            num_success = len(result[test_name][True])
        if False in result[test_name]:
            num_failure = len(result[test_name][False])
        print(f"{test_name}: {float(num_success) / (num_success + num_failure)}% success")

    return result


def create_shifted_pos(df: pd.DataFrame):
    df['prior ' + round_id_column] = df[round_id_column].shift(periods=-1)
    for player in specific_player_place_area_columns:
        df['prior ' + player.player_id] = df[player.player_id].shift(periods=-1)
        for pos in player.pos:
            df['prior ' + pos] = df[pos].shift(periods=-1)


cmap = mpl.cm.get_cmap("Set3").colors


def plot_similar_rounds(learned: bool, test_name: str, success: bool, round_ids: List[int], df: pd.DataFrame):
    with Image.open(d2_radar_path) as im:
        fig = plt.figure(figsize=(8,8), dpi=256)
        ax = plt.axes()
        ax.imshow(im)
        for round_id in round_ids:
            round_df = df[df[round_id_column] == round_id]
            for _, tick in round_df.iterrows():
                player_id = 0
                for player in specific_player_place_area_columns:
                    player_id += 1
                    if tick[player.player_id] == -1 or tick[player.player_id] != tick['prior ' + player.player_id] or \
                            tick[round_id_column] != tick['prior ' + round_id_column]:
                        continue
                    cur_pos = MapCoordinate(tick[player.pos[0]], tick[player.pos[1]], tick[player.pos[2]])
                    cur_pos_canvas = cur_pos.get_canvas_coordinates()
                    prior_pos = MapCoordinate(tick['prior ' + player.pos[0]], tick['prior ' + player.pos[1]],
                                              tick['prior ' + player.pos[2]])
                    prior_pos_canvas = prior_pos.get_canvas_coordinates()

                    color = cmap[player_id]
                    ax.plot([prior_pos_canvas.x, cur_pos_canvas.x], [prior_pos_canvas.y, cur_pos_canvas.y], color=color,
                            alpha=0.025, linestyle='-', linewidth=1)
        plt.axis('off')
        plt.savefig(plots_path / f"{'learned' if learned else 'handcraft'}_{test_name}_{success}.png", bbox_inches='tight', pad_inches=0)


def prepare_df(p: Path) -> Tuple[pd.DataFrame, RoundsPerTestResult]:
    df = load_hdf5_to_pd(p)
    df = df.copy()
    df[test_name_col] = df[test_name_col].str.decode("utf-8")
    rounds_per_test_result = get_rounds_per_test_result(df)
    create_shifted_pos(df)
    return df, rounds_per_test_result


if __name__ == "__main__":
    os.makedirs(plots_path, exist_ok=True)

    for p, learned in zip([manual_latent_team_hdf5_data_path, rollout_latent_team_hdf5_data_path], [False, True]):
        df, rounds_per_test_result = prepare_df(p)

        for test_name, success_to_rounds in rounds_per_test_result.items():
            print(f"processing {test_name}")
            for success, rounds in success_to_rounds.items():
                plot_similar_rounds(learned, test_name, success, rounds, df)
