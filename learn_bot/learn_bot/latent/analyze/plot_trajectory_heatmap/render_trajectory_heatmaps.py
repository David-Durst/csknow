from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from learn_bot.libs.pil_helpers import concat_horizontal, concat_vertical, concat_horizontal_vertical_with_extra, \
    repeated_paste_horizontal
from learn_bot.latent.analyze.compare_trajectories.plot_trajectories_and_events import title_font
from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmaps import get_title_to_line_buffers, \
    get_title_to_team_to_key_event_pos, get_debug_event_counting, get_title_to_key_events, get_title_to_num_points
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions
from learn_bot.latent.analyze.test_traces.run_trace_visualization import d2_img, \
    bot_ct_color_list, bot_t_color_list

scale_factor = 0
max_value = 0

a_example_path = Path(__file__).parent / '..' / 'a_site_example.jpg'
b_example_path = Path(__file__).parent / '..' / 'b_site_example.jpg'

alpha_color = False
tab20b_cmap = plt.get_cmap('tab20b')
tab20b_np = np.asarray([[c[0] * 255, c[1] * 255, c[2] * 255, 255] for c in tab20b_cmap.colors])

def plot_one_image_one_team(title: str, ct_team: bool, team_color: List, saturated_team_color: List,
                            base_img: Image.Image, custom_buffer: Optional[np.ndarray] = None):

    title_to_buffers = get_title_to_line_buffers()
    title_to_num_points = get_title_to_num_points()

    if custom_buffer is None:
        buffer = title_to_buffers[title].get_buffer(ct_team)
    else:
        buffer = custom_buffer

    if alpha_color:
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
    else:
        buffer_repeated = buffer[:, :, np.newaxis].repeat(4, axis=2)

        if max_value > 255:
            # one extra subtracted from numerator so that all values scaled to 0-19 in index lookup
            above_saturated_scaled_buffer = (buffer - 256) / (max_value - 255)
        else:
            above_saturated_scaled_buffer = buffer
        limit_scaled_buffer = np.where(buffer <= 255, buffer / 16, 16 + above_saturated_scaled_buffer * 4)
        uint_limit_scaled_buffer = np.uint8(limit_scaled_buffer)
        color_buffer_with_bad_zeros = tab20b_np[uint_limit_scaled_buffer]
        transparent_color_buffer = color_buffer_with_bad_zeros.copy()
        transparent_color_buffer[:, :, 3] = 0
        color_buffer = np.where(buffer_repeated == 0., transparent_color_buffer, color_buffer_with_bad_zeros)

    uint8_color_buffer = np.uint8(color_buffer)

    base_img.alpha_composite(Image.fromarray(uint8_color_buffer, 'RGBA'))

    #title_drw = ImageDraw.Draw(base_img)
    #if custom_buffer is None:
    #    title_text = title + f", \n Num Points Both Teams {title_to_num_points[title]} Scale Factor {scale_factor}"
    #else:
    #    title_text = title
    #_, _, w, h = title_drw.textbbox((0, 0), title_text, font=title_font)
    #title_drw.text(((d2_img.width - w) / 2, (d2_img.height * 0.1 - h) / 2),
    #               title_text, fill=(255, 255, 255, 255), font=title_font)


def scale_buffers_by_points(titles: List[str]):
    global scale_factor, max_value

    title_to_buffers = get_title_to_line_buffers()
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
    ct_title_images: List[Image.Image] = []
    t_title_images: List[Image.Image] = []
    scale_buffers_by_points(titles)

    for title in titles:
        if plot_teams_separately:
            images_per_title: List[Image.Image] = []
            # image with just ct
            base_ct_d2_img = d2_img.copy().convert("RGBA")
            plot_one_image_one_team(title, True, bot_ct_color_list, saturated_ct_color_list, base_ct_d2_img)
            base_ct_d2_img.thumbnail([1000, 1000], Image.ANTIALIAS)
            images_per_title.append(base_ct_d2_img)
            ct_title_images.append(base_ct_d2_img)

            # image with just t
            base_t_d2_img = d2_img.copy().convert("RGBA")
            plot_one_image_one_team(title, False, bot_t_color_list, saturated_t_color_list, base_t_d2_img)
            base_t_d2_img.thumbnail([1000, 1000], Image.ANTIALIAS)
            images_per_title.append(base_t_d2_img)
            t_title_images.append(base_t_d2_img)
            title_images.append(concat_horizontal(images_per_title))
        else:
            # image with everyone
            base_both_d2_img = d2_img.copy().convert("RGBA")
            plot_one_image_one_team(title, True, bot_ct_color_list, saturated_ct_color_list, base_both_d2_img)
            plot_one_image_one_team(title, False, bot_t_color_list, saturated_t_color_list, base_both_d2_img)
            title_images.append(base_both_d2_img)

    if len(titles) == 4 and trajectory_filter_options.is_no_filter():
        extra_width_spacing = 125 + 35
        extra_height_spacing = 175 + 76
        complete_image_with_highlights = concat_horizontal_vertical_with_extra(ct_title_images,
                                                                               t_title_images,
                                                                               #in_game_image_height +
                                                                               # height spacing for in-engine images
                                                                               2 * extra_height_spacing, #+ extra_height_for_highlights + ,
                                                                               extra_width_spacing,
                                                                               extra_height_spacing, 83 + 35, 83 + 76)
        ct_undera_focus_ims: List[Image.Image] = []
        ct_b_focus_ims: List[Image.Image] = []
        #ct_mid_focus_ims: List[Image.Image] = []
        for im in ct_title_images:
            # undera
            ct_undera_focus_im = im.crop((663, 182, 842, 270))
            ct_undera_focus_im = ct_undera_focus_im.resize((481, 237), Image.ANTIALIAS)
            ct_undera_focus_ims.append(ct_undera_focus_im)

            ct_b_focus_im = im.crop((93, 112, 251, 325))
            ct_b_focus_im = ct_b_focus_im.resize((307, 413), Image.ANTIALIAS)
            ct_b_focus_ims.append(ct_b_focus_im)

            #ct_mid_focus_im = im.crop((468, 395, 493, 422))
            #ct_mid_focus_im = ct_mid_focus_im.resize((98, 100), Image.ANTIALIAS)
            #ct_mid_focus_ims.append(ct_mid_focus_im)
        repeated_paste_horizontal(complete_image_with_highlights, ct_b_focus_ims, 10, 728 + 76, 1000 + extra_width_spacing)
        undera_top = 728 + 76 + 413 - 237
        repeated_paste_horizontal(complete_image_with_highlights, ct_undera_focus_ims, 548, undera_top, 1000 + extra_width_spacing)
        #repeated_paste_horizontal(complete_image_with_highlights, ct_mid_focus_ims, 348, 904, 1000 + extra_width_spacing)

        #a_example_im = Image.open(a_example_path)
        #a_example_im = a_example_im.resize((in_game_image_width, in_game_image_height), Image.Resampling.LANCZOS)
        #complete_image_with_highlights.paste(a_example_im, (1000 + extra_width_spacing // 2 - in_game_image_width // 2,
        #                                                    2000 + 2 * extra_height_spacing + extra_height_for_highlights))

        t_b_focus_ims: List[Image.Image] = []
        t_longa_focus_ims: List[Image.Image] = []
        for im in t_title_images:
            t_b_focus_im = im.crop((101, 141, 250, 363))
            t_b_focus_im = t_b_focus_im.resize((263, 372), Image.ANTIALIAS)
            t_b_focus_ims.append(t_b_focus_im)

            t_longa_focus_im = im.crop((825, 482, 945, 624))
            t_longa_focus_im = t_longa_focus_im.resize((292, 345), Image.ANTIALIAS)
            t_longa_focus_ims.append(t_longa_focus_im)
        repeated_paste_horizontal(complete_image_with_highlights, t_b_focus_ims, 10, 1000 + extra_height_spacing + 704, 1000 + extra_width_spacing)
        longa_top = 1000 + extra_height_spacing + 704 + 372 - 345
        repeated_paste_horizontal(complete_image_with_highlights, t_longa_focus_ims, 737, longa_top, 1000 + extra_width_spacing)

        #b_example_im = Image.open(b_example_path)
        #b_example_im = b_example_im.resize((in_game_image_width, in_game_image_height), Image.Resampling.LANCZOS)
        #complete_image_with_highlights.paste(b_example_im, (3 * (1000 + extra_width_spacing) - extra_width_spacing // 2 - in_game_image_width // 2,
        #                                                    2000 + 2 * extra_height_spacing + extra_height_for_highlights))
        complete_image_with_highlights.save(plots_path / 'complete_with_highlights.png')
        get_colorbar_image(plots_path)

    complete_image = concat_vertical(title_images)
    complete_image.save(plots_path / (str(trajectory_filter_options) + '.png'))


def get_colorbar_image(plots_path: Path):
    a = np.array([[0, 1]])
    fig = plt.figure(figsize=(0.481, 3.217))
    img = plt.imshow(a, cmap="tab20b")
    plt.gca().set_visible(False)
    cax = plt.axes([0.2, 0.1, 0.4, 0.8])
    cbar = plt.colorbar(orientation="vertical", cax=cax)
    cbar.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y_tick_labels = [str(int(i * 255 / scale_factor)) for i in [0, 0.2, 0.4, 0.6, 0.8]]
    y_tick_labels.append(f"> {y_tick_labels[-1]}")
    #y_tick_labels.append(str(max_value))
    cbar.ax.set_yticklabels(y_tick_labels, ha='right')
    cbar.ax.tick_params(axis="y", labelsize=8, pad=16)
    cbar.ax.set_ylabel("Points", rotation=270, labelpad=10, fontsize=8)
    plt.savefig(plots_path / "colorbar.pdf", dpi=300)



