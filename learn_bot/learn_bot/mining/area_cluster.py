from pathlib import Path

import pandas as pd

from learn_bot.latent.place_area.column_names import *
from learn_bot.latent.train import latent_team_hdf5_data_path
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image, ImageDraw


d2_top_left_x = -2476
d2_top_left_y = 3239
defaultCanvasSize = 700
bigCanvasSize = 2048
minimapWidth = 2048
minimapHeight = 2048
minimapScale = 4.4 * 1024 / minimapHeight
d2_radar_path = Path(__file__).parent / '..' / '..' / '..' / 'web_vis' / 'vis_images' / 'de_dust2_radar_upsampled_all_labels.png'


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __str__(self) -> str:
        return f'''({self.x}, {self.y}, {self.z})'''


class MapCoordinate:
    coords: Vec3

    def __init__(self, x: float, y: float, z: float, from_canvas_pixels: bool = False):
        if from_canvas_pixels:
            pctX = x / minimapWidth
            x = d2_top_left_x + minimapScale * minimapWidth * pctX
            pctY = y / minimapHeight
            y = d2_top_left_y - minimapScale * minimapHeight * pctY
        self.coords = Vec3(x, y, z)

    def get_canvas_coordinates(self) -> Vec3:
        return Vec3(
            (self.coords.x - d2_top_left_x) / minimapScale,
            (d2_top_left_y - self.coords.y) / minimapScale,
            self.coords.z
        )

    def draw(self, im_draw: ImageDraw, fill=(255, 0, 0, 255), outline=(255, 0, 0, 255)):
        canvas_coords = self.get_canvas_coordinates()
        x_min = canvas_coords.x - 20
        y_min = canvas_coords.y - 20
        x_max = canvas_coords.x + 20
        y_max = canvas_coords.y + 20
        im_draw.rectangle([x_min, y_min, x_max, y_max], fill=fill, outline=outline)


def cluster_one_team(team_data_df: pd.DataFrame, player_place_area_columns: List[PlayerPlaceAreaColumns], team_str: str, planted_a: bool):
    team_data_df = team_data_df[team_data_df[c4_status_col] == (0 if planted_a else 1)]
    player_pos_arrs: List[np.ndarray] = []
    for player_cols in player_place_area_columns:
        single_player_df = team_data_df.loc[:, player_cols.pos + player_cols.vel + [player_cols.ct_team, player_cols.player_id]]
        single_player_df['vel_magnitude'] = (
                single_player_df[player_cols.vel[0]] ** 2 +
                single_player_df[player_cols.vel[1]] ** 2 +
                single_player_df[player_cols.vel[2]] ** 2).pow(1 / 2)
        invalid_rows = single_player_df[(single_player_df[player_cols.player_id] == -1) &
                                        (single_player_df[player_cols.pos[0]] < 5000.)]
        if len(invalid_rows) > 0:
            print("BADDD")
        single_player_df = single_player_df[(single_player_df['vel_magnitude'] < 1.) &
                                            (single_player_df[player_cols.player_id] != -1)]
        player_pos_arrs.append(single_player_df.loc[:, player_cols.pos].to_numpy())
        (team_data_df.loc[:, player_cols.vel].to_numpy())
    player_pos_arr = np.concatenate(player_pos_arrs)
    pos_kmeans = KMeans(n_clusters=50, random_state=0, n_init="auto").fit(player_pos_arr)
    # print(pos_kmeans.cluster_centers_)
    with Image.open(d2_radar_path) as im:
        d2_draw = ImageDraw.Draw(im)
        for i in range(len(pos_kmeans.cluster_centers_)):
            coord_arr = pos_kmeans.cluster_centers_[i]
            MapCoordinate(coord_arr[0], coord_arr[1], coord_arr[2]).draw(d2_draw)
        im.save(Path(__file__).parent / ("de_dust2_heatmap_" + team_str + "_" + ("a" if planted_a else "b") + ".png"))


def area_cluster():
    team_data_df = load_hdf5_to_pd(latent_team_hdf5_data_path)
    team_data_df = team_data_df[(team_data_df['valid'] == 1.) & (team_data_df['c4 status'] < 2) &
                                (team_data_df['retake save round tick'] == 0)]# & (team_data_df['round id'] == 14)]
    #print(team_data_df["distribution nearest place 0 T 0"].value_counts())
    for team_str in team_strs:
        player_place_area_columns = [PlayerPlaceAreaColumns(team_str, player_index)
                                     for player_index in range(max_enemies)]
        cluster_one_team(team_data_df, player_place_area_columns, team_str, True)
        cluster_one_team(team_data_df, player_place_area_columns, team_str, False)




if __name__ == "__main__":
    area_cluster()
