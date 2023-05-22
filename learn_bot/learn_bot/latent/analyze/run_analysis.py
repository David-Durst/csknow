from pathlib import Path

import pandas as pd
import os
from typing import Dict, List

from PIL import Image
from PIL.ImageDraw import ImageDraw

from learn_bot.engagement_aim.analysis.fitts import test_name_col
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.place_area.column_names import test_success_col
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.latent.train import manual_latent_team_hdf5_data_path
from learn_bot.mining.area_cluster import d2_radar_path

plots_path = Path(__file__).parent / 'plots'


def get_rounds_per_test_result(data_df: pd.DataFrame) -> Dict[str, Dict[bool, List[int]]]:
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
            num_failure = len(result[test_name][False])
        print(f"{test_name}: {float(success) / (num_success + num_failure)}% success")

    return result


def plot_similar_rounds(human: bool, test_name: str, success: bool, round_ids: List[int], df: pd.DataFrame):
    with Image.open(d2_radar_path) as im:
        d2_draw = ImageDraw.Draw(im)
        for round_id in round_ids:
            round_df = df[df[round_id_column] == round_id]
            round_df['shifted_pos']
            coord_arr = pos_kmeans.cluster_centers_[i]
            MapCoordinate(coord_arr[0], coord_arr[1], coord_arr[2]).draw(d2_draw)

        im.save(plots_path / f"{'human' if human else 'bot'}_{test_name}_{success}.png")


if __name__ == "__main__":
    os.makedirs(plots_path, exist_ok=True)

    all_data_df = load_hdf5_to_pd(manual_latent_team_hdf5_data_path)
    all_data_df = all_data_df.copy()
    all_data_df[test_name_col] = all_data_df[test_name_col].str.decode("utf-8")
    rounds_per_test_result = get_rounds_per_test_result(all_data_df)
