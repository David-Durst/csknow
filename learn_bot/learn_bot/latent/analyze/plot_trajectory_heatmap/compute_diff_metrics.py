from dataclasses import dataclass
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
plot_flow = False
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


@dataclass
class DiffMetrics:
    emd_value: float
    partial_emd_value: float
    total_variation_distance: float
    kl_divergence: float
    symmetric_kl_divergence: float
    num_points: float
    num_baseline_points: float
    num_points_relative_to_baseline: float

    def __str__(self) -> str:
        return f"EMD {self.emd_value:.2f} PEMD {self.partial_emd_value:.2f} " \
               f"TV {self.total_variation_distance:.2f} " \
               f"KLD {self.kl_divergence:.2f} SKLD {self.symmetric_kl_divergence:.2f} " \
               f"Ratio Points {self.num_points_relative_to_baseline:.3f}"

    def csv(self) -> str:
        return f"{self.emd_value}, {self.partial_emd_value}, " \
               f"{self.total_variation_distance}, " \
               f"{self.kl_divergence}, {self.symmetric_kl_divergence:}, " \
               f"{self.num_points_relative_to_baseline}"


def sum_diff_metrics(diff_metrics_list: List[DiffMetrics]) -> DiffMetrics:
    result = DiffMetrics(emd_value=0, partial_emd_value=0, total_variation_distance=0, kl_divergence=0,
                         symmetric_kl_divergence=0, num_points=0, num_baseline_points=0,
                         num_points_relative_to_baseline=0)
    for dm in diff_metrics_list:
        result.emd_value += dm.emd_value
        result.partial_emd_value += dm.partial_emd_value
        result.total_variation_distance += dm.total_variation_distance
        result.kl_divergence += dm.kl_divergence
        result.symmetric_kl_divergence += dm.symmetric_kl_divergence
        result.num_points += dm.num_points
        result.num_baseline_points += dm.num_baseline_points

    result.num_points_relative_to_baseline = result.num_points / result.num_baseline_points
    return result


def compute_kld(p: np.ndarray, q: np.ndarray) -> float:
    pq_nonzero_coordinates = (p * q).nonzero()
    q_nonzero = q[pq_nonzero_coordinates[0], pq_nonzero_coordinates[1]]
    p_nonzero = p[pq_nonzero_coordinates[0], pq_nonzero_coordinates[1]]
    if np.min(q_nonzero) == 0.:
        print('bad')
    return float(np.sum(p_nonzero * np.log(p_nonzero / q_nonzero)))


# compute emd with weights scaled to dist_b, treating that as baseline
def compute_one_distribution_pair_metrics(dist_a: ImageBuffers, dist_b: ImageBuffers,
                                          title_a: str, title_b: str,
                                          model_team_buffers: Dict[str, List[np.ndarray]],
                                          model_team_metrics: Dict[str, List[DiffMetrics]],
                                          model_team_flow: Dict[str, List[Image.Image]]):
    if title_a in model_team_buffers:
        raise Exception("running diff with same source metric twice")
    model_team_buffers[title_a] = []
    model_team_metrics[title_a] = []
    if plot_flow:
        model_team_flow[title_a] = []
    record_b = False
    # don't record a title that was already recorded, as if b used multiple times as baseline
    if title_b not in model_team_buffers:
        record_b = True
        model_team_buffers[title_b] = []

    for ct_team in [True, False]:
        # downsample so reasonable size, get x,y coordinates and weights of clusters with non-zero weight
        a_buffer = dist_a.get_buffer(ct_team)
        a_buffer_downsampled = block_reduce(a_buffer, (20, 20), np.sum)
        if plot_downsampled:
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
        a_sum = np.sum(a_non_zero_values)
        b_sum = np.sum(b_non_zero_values)
        scaled_a_non_zero_values = a_non_zero_values / a_sum
        scaled_b_non_zero_values = b_non_zero_values / b_sum
        # scaled so that one is max 1, other is less than 1, enable partial wasserstein
        partial_scaled_a_non_zero_values = a_non_zero_values / max(a_sum, b_sum)
        partial_scaled_b_non_zero_values = b_non_zero_values / max(a_sum, b_sum)

        if plot_scaled:
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
        dist_matrix = ot.dist(a_non_zero_coords_np, b_non_zero_coords_np, metric='euclidean')

        # compute emd
        emd_matrix, emd_dict = ot.emd(scaled_a_non_zero_values, scaled_b_non_zero_values,
                                      dist_matrix, log=True, numItermax=1000000)
        partial_emd_matrix, partial_emd_dict = ot.partial.partial_wasserstein(partial_scaled_a_non_zero_values,
                                                                              partial_scaled_b_non_zero_values,
                                                                              dist_matrix, log=True, numItermax=1000000)
        total_variation = float(np.sum(np.abs(a_buffer - b_buffer)))
        kl_divergence = compute_kld(a_buffer, b_buffer)
        symmetric_kl_divergence = kl_divergence + compute_kld(b_buffer, a_buffer)
        model_team_metrics[title_a].append(DiffMetrics(emd_dict['cost'], partial_emd_dict['cost'],
                                                       total_variation, kl_divergence, symmetric_kl_divergence,
                                                       float(a_sum), float(b_sum), a_sum / b_sum))
        if plot_flow:
            fig = plt.figure(figsize=(4,4))
            ot.plot.plot2D_samples_mat(a_non_zero_coords_np[:, [1, 0]], b_non_zero_coords_np[:, [1, 0]],
                                       emd_matrix, c=[.5, .5, 1])
            plt.gca().invert_yaxis()
            fig.canvas.draw()
            model_team_flow[title_a].append(
                Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))


title_font = ImageFont.truetype("arial.ttf", 12)


def plot_emd_buffer(model_team_buffers: Dict[str, List[np.ndarray]], model_team_metrics: Dict[str, List[DiffMetrics]],
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
            team_img = Image.fromarray(scaled_buffer, 'L').resize((800, 800), Image.NEAREST)
            team_img_drw = ImageDraw.Draw(team_img)
            img_text = title[:60]
            if title in model_team_metrics:
                img_text += " \n " + str(model_team_metrics[title][i])
            _, _, w, h = team_img_drw.textbbox((0, 0), img_text, font=title_font)
            team_img_drw.text(((team_img.width - w) / 2, (team_img.height * 0.1 - h) / 2),
                          img_text, fill=(255), font=title_font)
            team_imgs.append(team_img)
        model_team_imgs.append(concat_horizontal(team_imgs))
    emd_img = concat_vertical(model_team_imgs)
    emd_img.save(plots_path / 'diff' / ('emd_' + str(trajectory_filter_options) + '.png'))


def plot_emd_flow(model_team_flow: Dict[str, List[Image.Image]], model_team_metrics: Dict[str, List[DiffMetrics]],
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
            if title in model_team_metrics:
                img_text += " \n " + str(model_team_metrics[title][i])
            _, _, w, h = team_img_drw.textbbox((0, 0), img_text, font=title_font)
            team_img_drw.text(((team_img.width - w) / 2, (team_img.height * 0.1 - h) / 2),
                              img_text, fill=(0, 0, 0, 255), font=title_font)
        model_team_imgs.append(concat_horizontal(team_imgs))
    emd_img = concat_vertical(model_team_imgs)
    emd_img.save(plots_path / 'diff' / ('emd_' + str(trajectory_filter_options) + '_flow.png'))


def compute_diff_metrics(titles: List[str], diff_indices: List[int], plots_path: Path,
                         trajectory_filter_options: TrajectoryFilterOptions):
    print(f'Computing earth movers distance for {", ".join(titles)}')
    title_to_buffers = get_title_to_buffers()
    #title_to_num_trajectory_ids = get_title_to_num_trajectory_ids()

    model_team_buffers: Dict[str, List[np.ndarray]] = {}
    model_team_metrics: Dict[str, List[DiffMetrics]] = {}
    model_team_flow: Dict[str, List[Image.Image]] = {}

    with tqdm(total=len(titles) - 1, disable=False) as pbar:
        for i, title in enumerate(titles[1:]):
            diff_title = titles[diff_indices[i]]
            compute_one_distribution_pair_metrics(title_to_buffers[title], title_to_buffers[diff_title],
                                                  title, diff_title, model_team_buffers, model_team_metrics,
                                                  model_team_flow)
            pbar.update(1)

    plot_emd_buffer(model_team_buffers, model_team_metrics, titles, plots_path, trajectory_filter_options)
    plot_emd_flow(model_team_flow, model_team_metrics, titles, plots_path, trajectory_filter_options)

    with open(plots_path / 'diff' / ('emd_' + str(trajectory_filter_options) + '.txt'), 'w') as f:
        f.write("title, emd")
        for title, team_metrics in model_team_metrics.items():
            model_metrics = sum_diff_metrics(team_metrics)
            f.write(f'{title}, {model_metrics.csv()}\n')
