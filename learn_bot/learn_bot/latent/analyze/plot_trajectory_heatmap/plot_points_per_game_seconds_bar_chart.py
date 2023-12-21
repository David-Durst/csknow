from pathlib import Path
from typing import List, Dict, Set

import pandas as pd

from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import get_title_to_num_points

data_points_per_game_seconds: List[Dict] = []
data_set_titles: Set = set()


def reset_points_per_game_seconds_state():
    global data_points_per_game_seconds, data_set_titles
    data_points_per_game_seconds = []
    data_set_titles = set()


game_seconds_col = 'Game Seconds'
data_set_title_col = 'Data Set'
num_points_col = 'Number of Points'


def record_points_per_one_game_seconds_range(round_game_seconds: range):
    data_points_per_game_seconds_row = {
        game_seconds_col: f"{round_game_seconds.start}-{round_game_seconds.stop}"
    }
    for data_set_title, num_points in get_title_to_num_points().items():
        data_points_per_game_seconds_row[data_set_title] = num_points
        data_set_titles.add(data_set_title)
    data_points_per_game_seconds.append(data_points_per_game_seconds_row)


def plot_points_per_game_seconds(plots_path: Path):
    data_points_per_game_seconds_df = pd.DataFrame.from_records(data_points_per_game_seconds)
    ax = data_points_per_game_seconds_df.plot(x=game_seconds_col, y=list(data_set_titles), kind='bar', rot=0)
    ax.get_figure().savefig(plots_path / 'data_points_by_time.png')
