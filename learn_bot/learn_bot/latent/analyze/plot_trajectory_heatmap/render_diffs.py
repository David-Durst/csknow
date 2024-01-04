from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_from_comparison import concat_horizontal, \
    concat_vertical
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import get_title_to_line_buffers
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.plot_trajectory_heatmap.render_trajectory_heatmaps import plot_one_image_one_team
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img

base_positive_color_list = [31, 210, 93, 0]
saturated_positive_color_list = [17, 162, 144, 0]
base_negative_color_list = [242, 48, 48, 0]
saturated_negative_color_list = [162, 7, 146, 0]


def plot_diff_one_image_one_team(title0: str, title1: str, ct_team: bool) -> Image.Image:
    title_to_buffers = get_title_to_line_buffers()
    buffer0 = title_to_buffers[title0].get_buffer(ct_team)
    buffer1 = title_to_buffers[title1].get_buffer(ct_team)
    delta_buffer = buffer0 - buffer1

    # split delta_buffer into two sub buffers (one positive, one negative), plot them independently
    positive_delta_buffer = np.where(delta_buffer >= 0, delta_buffer, 0)
    negative_delta_buffer = np.where(delta_buffer < 0, -1 * delta_buffer, 0)

    base_green_img = Image.new("RGBA", d2_img.size, (0, 0, 0, 255))
    ct_team_str = 'CT' if ct_team else 'T'
    plot_one_image_one_team(f"{title0} (Green) vs {title1} (Red) {ct_team_str}", ct_team, base_positive_color_list,
                            saturated_positive_color_list, base_green_img, positive_delta_buffer)
    green_np = np.asarray(base_green_img)
    base_red_img = Image.new("RGBA", d2_img.size, (0, 0, 0, 255))
    plot_one_image_one_team(f"{title0} (Green) vs {title1} (Red) {ct_team_str}", ct_team, base_negative_color_list,
                            saturated_negative_color_list, base_red_img, negative_delta_buffer)
    red_np = np.asarray(base_red_img)

    combined_np = red_np + green_np
    #base_img = d2_img.copy().convert("RGBA")
    #base_img.alpha_composite(Image.fromarray(combined_np, 'RGBA'))
    #return base_img
    return Image.fromarray(combined_np, 'RGBA')


def plot_trajectory_diffs_to_image(titles: List[str], diff_indices: List[int], plots_path: Path,
                                   trajectory_filter_options: TrajectoryFilterOptions):
    # assuming already called plot_trajectories_to_image, so already did scaling
    title_images: List[Image.Image] = []

    for i, title in enumerate(titles[1:]):
        images_per_title: List[Image.Image] = []

        images_per_title.append(plot_diff_one_image_one_team(title, titles[diff_indices[i]], True))
        images_per_title.append(plot_diff_one_image_one_team(title, titles[diff_indices[i]], False))

        title_images.append(concat_horizontal(images_per_title))

    complete_image = concat_vertical(title_images)
    complete_image.save(plots_path / 'diff' / (str(trajectory_filter_options) + '.png'))
