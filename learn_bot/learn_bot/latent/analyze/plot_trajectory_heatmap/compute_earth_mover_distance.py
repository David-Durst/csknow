from pathlib import Path
from typing import List, Dict

import numpy as np
import ot
from PIL import Image, ImageFont, ImageDraw
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage.measure import block_reduce
from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_from_comparison import concat_horizontal, \
    concat_vertical
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import get_title_to_buffers, ImageBuffers, \
    get_title_to_num_trajectory_ids
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions

# https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/RUBNER/emd.htm
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

# compute emd with weights scaled to dist_b, treating that as baseline
def compute_one_earth_mover_distance(dist_a: ImageBuffers, dist_b: ImageBuffers,
                                     title_a: str, title_b: str,
                                     model_team_buffers: Dict[str, List[np.ndarray]],
                                     model_team_emd: Dict[str, List[float]]) -> float:
    # don't record a title that was already recorded, as if b used multiple times as baseline
    record_a = False
    if title_a not in model_team_buffers:
        record_a = True
        model_team_buffers[title_a] = []
        model_team_emd[title_a] = []
    record_b = False
    if title_b not in model_team_buffers:
        record_b = True
        model_team_buffers[title_b] = []

    emd = 0
    for ct_team in [True, False]:
        # downsample so reasonable size, get x,y coordinates and weights of clusters with non-zero weight
        a_buffer = dist_a.get_buffer(ct_team)
        a_buffer_downsampled = block_reduce(a_buffer, (20, 20), np.sum)
        if record_a:
            model_team_buffers[title_a].append(a_buffer_downsampled)
        a_non_zero_coords = a_buffer_downsampled.nonzero()
        a_non_zero_coords_np = np.column_stack(a_non_zero_coords)
        a_non_zero_values = a_buffer_downsampled[a_non_zero_coords[0], a_non_zero_coords[1]].astype(np.float64)

        b_buffer = dist_b.get_buffer(ct_team)
        b_buffer_downsampled = block_reduce(b_buffer, (20, 20), np.sum)
        if record_b:
            model_team_buffers[title_b].append(b_buffer_downsampled)
        b_non_zero_coords = b_buffer_downsampled.nonzero()
        b_non_zero_coords_np = np.column_stack(b_non_zero_coords)
        b_non_zero_values = b_buffer_downsampled[b_non_zero_coords[0], b_non_zero_coords[1]].astype(np.float64)

        # scale a_non_zero_values sum to b_non_zero_values sum
        # while the inf.ed.ac.uk shows that EMD is defined if different sums, POT only supports weights that sum to 1
        scaled_a_non_zero_values = a_non_zero_values / np.sum(a_non_zero_values)
        scaled_b_non_zero_values = b_non_zero_values / np.sum(b_non_zero_values)

        # compute distance between all cluster coordinates
        dist_matrix = ot.dist(a_non_zero_coords_np, b_non_zero_coords_np)

        # compute emd
        new_emd = ot.emd2(scaled_a_non_zero_values, scaled_b_non_zero_values, dist_matrix, numItermax=1000000)
        emd += new_emd
        if record_a:
            model_team_emd[title_a].append(new_emd)
    return emd

title_font = ImageFont.truetype("arial.ttf", 12)

# first index is the model, second is the team
def plot_emd_buffer(model_team_buffers: Dict[str, List[np.ndarray]], model_team_emd: Dict[str, List[float]],
                    titles: List[str], plots_path: Path, trajectory_filter_options: TrajectoryFilterOptions):
    model_team_imgs: List[Image.Image] = []
    for title in titles:
        if title not in model_team_buffers:
            continue
        team_buffers = model_team_buffers[title]
        team_imgs = []
        for i, buffer in enumerate(team_buffers):
            max_value = np.max(buffer)
            scaled_buffer = np.uint8(buffer / max_value * 255)
            team_img = Image.fromarray(scaled_buffer, 'L').resize((500, 500), Image.NEAREST)
            team_img_drw = ImageDraw.Draw(team_img)
            img_text = title[:60]
            if title in model_team_emd:
                img_text += " " + str(f"{model_team_emd[title][i]:.2f}")
            _, _, w, h = team_img_drw.textbbox((0, 0), img_text, font=title_font)
            team_img_drw.text(((team_img.width - w) / 2, (team_img.height * 0.05 - h) / 2),
                          img_text, fill=(255), font=title_font)
            team_imgs.append(team_img)
        model_team_imgs.append(concat_horizontal(team_imgs))
    emd_img = concat_vertical(model_team_imgs)
    emd_img.save(plots_path / 'diff' / ('emd_' + str(trajectory_filter_options) + '.png'))


def compute_trajectory_earth_mover_distances(titles: List[str], diff_indices: List[int], plots_path: Path,
                                             trajectory_filter_options: TrajectoryFilterOptions):
    print(f'Computing earth movers distance for {", ".join(titles)}')
    title_to_buffers = get_title_to_buffers()
    #title_to_num_trajectory_ids = get_title_to_num_trajectory_ids()

    titles_to_emd: Dict[str, float] = {}
    model_team_buffers: Dict[str, List[np.ndarray]] = {}
    model_team_emd: Dict[str, List[float]] = {}

    with tqdm(total=len(titles) - 1, disable=False) as pbar:
        for i, title in enumerate(titles[1:]):
            diff_title = titles[diff_indices[i]]
            titles_to_emd[f'{title} vs {diff_title}'] = \
                compute_one_earth_mover_distance(title_to_buffers[title], title_to_buffers[diff_title],
                                                 title, diff_title, model_team_buffers, model_team_emd)
            pbar.update(1)

    plot_emd_buffer(model_team_buffers, model_team_emd, titles, plots_path, trajectory_filter_options)

    with open(plots_path / 'diff' / ('emd_' + str(trajectory_filter_options) + '.txt'), 'w') as f:
        for titles, emd in titles_to_emd.items():
            f.write(f'{titles}: {emd}\n')
