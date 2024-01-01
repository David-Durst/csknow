from pathlib import Path
from typing import List, Dict

import numpy as np
import ot
import ot.plot
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
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

plot_downsampled = False
plot_scaled = True
plot_flow = True
debug_printing = False


def print_halves(title: str, buffer: np.ndarray):
    if not debug_printing:
        return
    num_rows, num_cols = buffer.shape
    top = buffer[:num_rows // 2, :].sum()
    bottom = buffer[num_rows // 2:, :].sum()
    left = buffer[:, :num_cols // 2].sum()
    right = buffer[:, num_cols // 2:].sum()
    print(f"{title} top {top:.2f} bottom {bottom:.2f} left {left:.2f} right {right:.2f}")


# compute emd with weights scaled to dist_b, treating that as baseline
def compute_one_earth_mover_distance(dist_a: ImageBuffers, dist_b: ImageBuffers,
                                     title_a: str, title_b: str,
                                     model_team_buffers: Dict[str, List[np.ndarray]],
                                     model_team_emd: Dict[str, List[float]],
                                     model_team_flow: Dict[str, List[Image.Image]]) -> float:
    # don't record a title that was already recorded, as if b used multiple times as baseline
    record_a = False
    if title_a not in model_team_buffers:
        record_a = True
        model_team_buffers[title_a] = []
        model_team_emd[title_a] = []
        if plot_flow:
            model_team_flow[title_a] = []
    record_b = False
    if title_b not in model_team_buffers:
        record_b = True
        model_team_buffers[title_b] = []

    emd = 0
    for ct_team in [True, False]:
        # downsample so reasonable size, get x,y coordinates and weights of clusters with non-zero weight
        a_buffer = dist_a.get_buffer(ct_team)
        a_buffer_downsampled = block_reduce(a_buffer, (20, 20), np.sum)
        if record_a and plot_downsampled:
            model_team_buffers[title_a].append(a_buffer_downsampled)
            print_halves(title_a, a_buffer_downsampled)
        a_non_zero_coords = a_buffer_downsampled.nonzero()
        a_non_zero_coords_np = np.column_stack(a_non_zero_coords)
        a_non_zero_values = a_buffer_downsampled[a_non_zero_coords[0], a_non_zero_coords[1]].astype(np.float64)

        b_buffer = dist_b.get_buffer(ct_team)
        b_buffer_downsampled = block_reduce(b_buffer, (20, 20), np.sum)
        if record_b and plot_downsampled:
            model_team_buffers[title_b].append(b_buffer_downsampled)
            print_halves(title_b, b_buffer_downsampled)
        b_non_zero_coords = b_buffer_downsampled.nonzero()
        b_non_zero_coords_np = np.column_stack(b_non_zero_coords)
        b_non_zero_values = b_buffer_downsampled[b_non_zero_coords[0], b_non_zero_coords[1]].astype(np.float64)

        # scale a_non_zero_values and b_non_zero_values to sum to 1
        # while the inf.ed.ac.uk shows that EMD is defined if different sums, POT only supports weights that sum to 1
        scaled_a_non_zero_values = a_non_zero_values / np.sum(a_non_zero_values)
        scaled_b_non_zero_values = b_non_zero_values / np.sum(b_non_zero_values)

        if record_a and plot_scaled:
            a_buffer_scaled = np.zeros(a_buffer_downsampled.shape, np.float)
            a_buffer_scaled[a_non_zero_coords[0], a_non_zero_coords[1]] = scaled_a_non_zero_values
            model_team_buffers[title_a].append(a_buffer_scaled)
            print_halves(title_a, a_buffer_scaled)
        if record_b and plot_scaled:
            b_buffer_scaled = np.zeros(b_buffer_downsampled.shape, np.float)
            b_buffer_scaled[b_non_zero_coords[0], b_non_zero_coords[1]] = scaled_b_non_zero_values
            model_team_buffers[title_b].append(b_buffer_scaled)
            print_halves(title_b, b_buffer_scaled)

        # compute distance between all cluster coordinates
        dist_matrix = ot.dist(a_non_zero_coords_np, b_non_zero_coords_np, metric='cityblock')

        # compute emd
        #new_emd = ot.emd2(scaled_a_non_zero_values, scaled_b_non_zero_values, dist_matrix, numItermax=1000000)
        #emd += new_emd
        emd_matrix, emd_dict = ot.emd(scaled_a_non_zero_values, scaled_b_non_zero_values, dist_matrix, log=True,
                                      numItermax=1000000)
        new_emd = emd_dict['cost']
        emd += new_emd
        if record_a:
            model_team_emd[title_a].append(new_emd)
            if plot_flow:
                fig = plt.figure(figsize=(4,4))
                ot.plot.plot2D_samples_mat(a_non_zero_coords_np[:, [1, 0]], b_non_zero_coords_np[:, [1, 0]],
                                           emd_matrix, c=[.5, .5, 1])
                plt.gca().invert_yaxis()
                fig.canvas.draw()
                model_team_flow[title_a].append(
                    Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
    return emd

title_font = ImageFont.truetype("arial.ttf", 12)


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


def plot_emd_flow(model_team_flow: Dict[str, List[Image.Image]], model_team_emd: Dict[str, List[float]],
                  titles: List[str], plots_path: Path, trajectory_filter_options: TrajectoryFilterOptions):
    if not plot_flow:
        return
    model_team_imgs: List[Image.Image] = []
    for title in titles:
        if title not in model_team_flow:
            continue
        team_imgs = model_team_flow[title]
        for i, team_img in enumerate(team_imgs):
            team_img_drw = ImageDraw.Draw(team_img)
            img_text = title[:50]
            if title in model_team_emd:
                img_text += " " + str(f"{model_team_emd[title][i]:.2f}")
            _, _, w, h = team_img_drw.textbbox((0, 0), img_text, font=title_font)
            team_img_drw.text(((team_img.width - w) / 2, (team_img.height * 0.05 - h) / 2),
                              img_text, fill=(0, 0, 0, 255), font=title_font)
        model_team_imgs.append(concat_horizontal(team_imgs))
    emd_img = concat_vertical(model_team_imgs)
    emd_img.save(plots_path / 'diff' / ('emd_' + str(trajectory_filter_options) + '_flow.png'))


def compute_trajectory_earth_mover_distances(titles: List[str], diff_indices: List[int], plots_path: Path,
                                             trajectory_filter_options: TrajectoryFilterOptions):
    print(f'Computing earth movers distance for {", ".join(titles)}')
    title_to_buffers = get_title_to_buffers()
    #title_to_num_trajectory_ids = get_title_to_num_trajectory_ids()

    titles_to_emd: Dict[str, float] = {}
    model_team_buffers: Dict[str, List[np.ndarray]] = {}
    model_team_emd: Dict[str, List[float]] = {}
    model_team_flow: Dict[str, List[Image.Image]] = {}

    with tqdm(total=len(titles) - 1, disable=False) as pbar:
        for i, title in enumerate(titles[1:]):
            diff_title = titles[diff_indices[i]]
            titles_to_emd[f'{title} vs {diff_title}'] = \
                compute_one_earth_mover_distance(title_to_buffers[title], title_to_buffers[diff_title],
                                                 title, diff_title, model_team_buffers, model_team_emd,
                                                 model_team_flow)
            pbar.update(1)

    plot_emd_buffer(model_team_buffers, model_team_emd, titles, plots_path, trajectory_filter_options)
    plot_emd_flow(model_team_flow, model_team_emd, titles, plots_path, trajectory_filter_options)

    with open(plots_path / 'diff' / ('emd_' + str(trajectory_filter_options) + '.txt'), 'w') as f:
        for titles, emd in titles_to_emd.items():
            f.write(f'{titles}: {emd}\n')
