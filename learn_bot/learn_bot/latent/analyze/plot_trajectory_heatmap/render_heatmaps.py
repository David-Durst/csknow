from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw

from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_from_comparison import concat_horizontal, \
    concat_vertical
from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_and_events import title_font
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import get_title_to_buffers, \
    get_title_to_team_to_key_event_pos, get_debug_event_counting, get_title_to_key_events, get_title_to_num_points
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, \
    bot_ct_color_list, bot_t_color_list

scale_factor = 0
max_value = 0


def plot_one_image_one_team(title: str, ct_team: bool, team_color: List, saturated_team_color: List,
                            base_img: Image.Image, custom_buffer: Optional[np.ndarray] = None):

    title_to_buffers = get_title_to_buffers()
    title_to_num_points = get_title_to_num_points()

    if custom_buffer is None:
        buffer = title_to_buffers[title].get_buffer(ct_team)
    else:
        buffer = custom_buffer
    color_buffer = buffer[:, :, np.newaxis].repeat(4, axis=2)
    color_buffer[:, :, 0] = team_color[0]
    color_buffer[:, :, 1] = team_color[1]
    color_buffer[:, :, 2] = team_color[2]

    # if saturate, then move to darker color to indicate
    saturated_color_buffer_entries = color_buffer[:, :, 3] >= 255
    if np.sum(saturated_color_buffer_entries) > 0:
        percent_saturated = color_buffer[saturated_color_buffer_entries][:, 3] / max_value
        full_alpha_team_color = np.asarray(team_color)
        full_alpha_team_color[3] = 255
        full_alpha_saturated_team_color = np.asarray(saturated_team_color)
        full_alpha_saturated_team_color[3] = 255
        color_buffer[saturated_color_buffer_entries] = \
            (percent_saturated[:, np.newaxis].repeat(4, axis=1) *
             full_alpha_saturated_team_color[np.newaxis, :].repeat(np.sum(saturated_color_buffer_entries), axis=0)) + \
            ((1 - percent_saturated[:, np.newaxis].repeat(4, axis=1)) *
             full_alpha_team_color[np.newaxis, :].repeat(np.sum(saturated_color_buffer_entries), axis=0))
    uint8_color_buffer = np.uint8(color_buffer)
    base_img.alpha_composite(Image.fromarray(uint8_color_buffer, 'RGBA'))

    title_drw = ImageDraw.Draw(base_img)
    if custom_buffer is None:
        title_text = title + f", \n Num Points Both Teams {title_to_num_points[title]} Scale Factor {scale_factor}"
    else:
        title_text = title
    _, _, w, h = title_drw.textbbox((0, 0), title_text, font=title_font)
    title_drw.text(((d2_img.width - w) / 2, (d2_img.height * 0.1 - h) / 2),
                   title_text, fill=(255, 255, 255, 255), font=title_font)


def scale_buffers_by_points(titles: List[str]):
    global scale_factor, max_value

    title_to_buffers = get_title_to_buffers()
    title_to_num_points = get_title_to_num_points()

    max_points_per_title = 0
    for title in titles:
        max_points_per_title = max(max_points_per_title, title_to_num_points[title])
    if max_points_per_title > 1e6:
        scale_factor = 8
    elif max_points_per_title > 5e5:
        scale_factor = 11
    elif max_points_per_title > 1e5:
        scale_factor = 14
    elif max_points_per_title > 5e4:
        scale_factor = 19
    elif max_points_per_title > 1e4:
        scale_factor = 25
    else:
        scale_factor = 30
    #scale_factor = int(25. / log(2.2 + max_points_per_title / 1300, 10))
    # compute scaling factor for points
    #max_99_percentile = -1
    #for title in titles:
    #    ct_buffer = title_to_buffers[title].get_buffer(True)
    #    max_99_percentile = max(max_99_percentile, np.percentile(ct_buffer, 99))
    #    #print(f'ct_buffer percentiles: f{np.percentile(ct_buffer, [50, 90, 95, 99, 99.9, 99.99, 99.999, 99.9999])}')
    #    t_buffer = title_to_buffers[title].get_buffer(False)
    #    max_99_percentile = max(max_99_percentile, np.percentile(t_buffer, 99))
    #    #print(f't_buffer percentiles: f{np.percentile(ct_buffer, [50, 90, 95, 99, 99.9, 99.99, 99.999, 99.9999])}')
    #scale_factor = int(ceil(255 / max_99_percentile))

    # compute max value for color overflow
    max_value = -1
    for title in titles:
        ct_buffer = title_to_buffers[title].get_buffer(True)
        ct_buffer *= scale_factor
        max_value = max(max_value, np.max(ct_buffer))
        t_buffer = title_to_buffers[title].get_buffer(False)
        t_buffer *= scale_factor
        max_value = max(max_value, np.max(t_buffer))
        #print(f"{title} ct max value {np.max(ct_buffer)}, argmax {np.unravel_index(ct_buffer.argmax(), ct_buffer.shape)}")
        #print(f"{title} t max value {np.max(t_buffer)}, argmax {np.unravel_index(t_buffer.argmax(), t_buffer.shape)}")


saturated_ct_color_list = [19, 2, 178, 0]
saturated_t_color_list = [178, 69, 2, 0]


def plot_trajectories_to_image(titles: List[str], plot_teams_separately: bool, plots_path: Path,
                               trajectory_filter_options: TrajectoryFilterOptions):
    #print(f"max pixel value after scaling before clamp to 255 {max_value}")
    if get_debug_event_counting():
        title_to_team_to_key_event_pos = get_title_to_team_to_key_event_pos()
        title_to_key_events = get_title_to_key_events()
        print(str(trajectory_filter_options))
        print(title_to_key_events["Human"])
        print(len(title_to_team_to_key_event_pos["Human"][True][0]) +
              len(title_to_team_to_key_event_pos["Human"][False][0]))
        return

    title_images: List[Image.Image] = []
    scale_buffers_by_points(titles)

    for title in titles:
        images_per_title: List[Image.Image] = []

        # image with everyone
        base_both_d2_img = d2_img.copy().convert("RGBA")
        plot_one_image_one_team(title, True, bot_ct_color_list, saturated_ct_color_list, base_both_d2_img)
        plot_one_image_one_team(title, False, bot_t_color_list, saturated_t_color_list, base_both_d2_img)
        images_per_title.append(base_both_d2_img)

        if plot_teams_separately:
            # image with just ct
            base_ct_d2_img = d2_img.copy().convert("RGBA")
            plot_one_image_one_team(title, True, bot_ct_color_list, saturated_ct_color_list, base_ct_d2_img)
            images_per_title.append(base_ct_d2_img)

            # image with just t
            base_t_d2_img = d2_img.copy().convert("RGBA")
            plot_one_image_one_team(title, False, bot_t_color_list, saturated_t_color_list, base_t_d2_img)
            images_per_title.append(base_t_d2_img)
            title_images.append(concat_horizontal(images_per_title))
        else:
            title_images.append(images_per_title[0])

    complete_image = concat_vertical(title_images)
    complete_image.save(plots_path / (str(trajectory_filter_options) + '.png'))
