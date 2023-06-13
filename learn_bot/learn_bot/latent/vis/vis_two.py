from typing import Dict

import pandas as pd

from learn_bot.libs.df_grouping import make_index_column
from learn_bot.mining.area_cluster import *
from learn_bot.latent.vis.draw_inference import draw_all_players, minimapWidth, minimapHeight, \
    draw_player_connection_lines
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageDraw, ImageTk as itk
from learn_bot.latent.analyze.comparison_column_names import *
import matplotlib as mpl


@dataclass
class RolloutToManualRoundData:
    rollout_round_id: int
    manual_round_id: int
    similarity_row: pd.Series
    similarity_match_df: pd.DataFrame
    agent_mapping: Dict[int, int]


RolloutToManualDict = Dict[int, Dict[str, RolloutToManualRoundData]]

cmap = mpl.cm.get_cmap("Set3").colors

def vis_two(rollout_data_df: pd.DataFrame, rollout_pred_df: pd.DataFrame,
            manual_data_df: pd.DataFrame, manual_pred_df: pd.DataFrame, rollout_to_manual_dict: RolloutToManualDict):
    make_index_column(rollout_data_df)
    make_index_column(rollout_pred_df)
    make_index_column(manual_data_df)
    make_index_column(manual_pred_df)

    #This creates the main window of an application
    window = tk.Tk()
    window.title("Comparison Delta Position Model")
    window.resizable(width=True, height=True)
    window.configure(background='grey')

    your_font = font.nametofont("TkDefaultFont")  # Get default font value into Font object
    your_font.actual()

    img_frame = tk.Frame(window)
    img_frame.pack(pady=5)
    d2_img = Image.open(d2_radar_path)
    d2_img = d2_img.resize((minimapWidth, minimapHeight), Image.ANTIALIAS)
    rollout_d2_img_draw = d2_img.copy()
    rollout_d2_photo_img = itk.PhotoImage(rollout_d2_img_draw)
    rollout_img_label = tk.Label(img_frame, image=rollout_d2_photo_img)
    rollout_img_label.pack(side="left")
    manual_d2_img_draw = d2_img.copy()
    manual_d2_photo_img = itk.PhotoImage(manual_d2_img_draw)
    manual_img_label = tk.Label(img_frame, image=manual_d2_photo_img)
    manual_img_label.pack(side="left")

    rounds = rollout_data_df.loc[:, round_id_column].unique().tolist()
    cur_round: int = -1
    metric_types = ['Unconstrained DTW', 'Slope Constrained DTW', 'Percentile ADE']
    cur_metric_type: str = metric_types[0]
    cur_similarity_tick_index: int = -1
    rollout_selected_df: pd.DataFrame = rollout_data_df
    rollout_pred_selected_df: pd.DataFrame = rollout_pred_df
    manual_selected_df: pd.DataFrame = manual_data_df
    manual_pred_selected_df: pd.DataFrame = manual_pred_df
    similarity_match_df: Optional[pd.DataFrame] = None
    rollout_to_manual_round_data: Optional[RolloutToManualRoundData] = None
    draw_max: bool = True
    draw_overlap: bool = False

    def round_slider_changed(cur_round_index):
        nonlocal cur_round
        cur_round = rounds[int(cur_round_index)]
        change_round_metric_dependent_data()

    def round_back_clicked():
        cur_round_index = int(round_slider.get())
        if cur_round_index > 0:
            cur_round_index -= 1
            round_slider.set(cur_round_index)
            round_slider_changed(cur_round_index)

    def round_forward_clicked():
        cur_round_index = int(round_slider.get())
        if cur_round_index < len(rounds) - 1:
            cur_round_index += 1
            round_slider.set(cur_round_index)
            round_slider_changed(cur_round_index)

    def metric_slider_changed(cur_metric_index):
        nonlocal cur_metric_type
        cur_metric_type = metric_types[int(cur_metric_index)]
        change_round_metric_dependent_data()

    def metric_back_clicked():
        nonlocal cur_metric_type
        cur_metric_index = int(metric_slider.get())
        if cur_metric_index > 0:
            cur_metric_index -= 1
            cur_metric_type = metric_types[cur_metric_index]
            metric_slider.set(cur_metric_index)
            metric_slider_changed(cur_metric_index)

    def metric_forward_clicked():
        nonlocal cur_metric_type
        cur_metric_index = int(metric_slider.get())
        if cur_metric_index < len(metric_types) - 1:
            cur_metric_index += 1
            cur_metric_type = metric_types[cur_metric_index]
            metric_slider.set(cur_metric_index)
            metric_slider_changed(cur_metric_index)

    def tick_slider_changed(cur_similarity_tick_index_str):
        nonlocal cur_similarity_tick_index, d2_img, rollout_d2_img_draw, rollout_img_label, \
            manual_d2_img_draw, manual_img_label, draw_max, draw_overlap
        if len(rollout_selected_df) > 0:
            cur_similarity_tick_index = int(cur_similarity_tick_index_str)

            cur_rollout_index = similarity_match_df.iloc[cur_similarity_tick_index].loc[first_matched_index_col]
            cur_rollout_row = rollout_selected_df.iloc[cur_rollout_index]
            cur_rollout_pred_row = rollout_pred_selected_df.iloc[cur_rollout_index]
            cur_rollout_tick_id = cur_rollout_row.loc[tick_id_column]
            cur_rollout_game_tick_id = cur_rollout_row.loc[tick_id_column]

            cur_manual_index = similarity_match_df.iloc[cur_similarity_tick_index].loc[second_matched_index_col]
            cur_manual_row = manual_selected_df.iloc[cur_manual_index]
            cur_manual_pred_row = manual_pred_selected_df.iloc[cur_manual_index]
            cur_manual_round = cur_manual_row.loc[round_id_column]
            cur_manual_tick_id = cur_manual_row.loc[tick_id_column]
            cur_manual_game_tick_id = cur_manual_row.loc[game_tick_number_column]

            tick_id_text_var.set(f"Rollout Tick ID: {cur_rollout_tick_id}, Rollout Game Tick ID: {cur_rollout_game_tick_id}, "
                                 f"Manual Tick Id: {cur_manual_tick_id}, Manual Game Tick ID: {cur_manual_game_tick_id}")
            round_id_text_var.set(f"Rollout Round ID: {int(cur_round)}, Rollout Round Number: {cur_rollout_row.loc['round number']}, "
                                  f"Manual Round ID: {int(cur_manual_round)}, Manual Round Number: {cur_manual_row.loc['round number']}")
            metric_id_text_var.set(cur_metric_type)

            rollout_d2_img_copy = d2_img.copy().convert("RGBA")
            rollout_d2_overlay_im = Image.new("RGBA", rollout_d2_img_copy.size, (255, 255, 255, 0))
            rollout_d2_img_draw = ImageDraw.Draw(rollout_d2_overlay_im)
            manual_d2_img_copy = d2_img.copy().convert("RGBA")
            manual_d2_overlay_im = Image.new("RGBA", manual_d2_img_copy.size, (255, 255, 255, 0))
            manual_d2_img_draw = ImageDraw.Draw(manual_d2_overlay_im)

            players_to_draw_str = player_distributions_var.get()
            if players_to_draw_str == "*":
                rollout_players_to_draw = list(range(0, len(specific_player_place_area_columns)))
                manual_players_to_draw = rollout_players_to_draw
            else:
                rollout_players_to_draw = [int(p) for p in players_to_draw_str.split(";")[0].split(",")]
                manual_players_to_draw = [int(p) for p in players_to_draw_str.split(";")[1].split(",")]


            rollout_colors = {}
            manual_colors = {}
            for src, tgt in rollout_to_manual_round_data.agent_mapping.items():
                rollout_colors[src] = cmap[src]
                manual_colors[tgt] = cmap[src]
                rollout_colors[src] = (int(255 * rollout_colors[src][0]), int(255 * rollout_colors[src][1]),
                                       int(255 * rollout_colors[src][2]))
                manual_colors[tgt] = (int(255 * manual_colors[tgt][0]), int(255 * manual_colors[tgt][1]),
                                      int(255 * manual_colors[tgt][2]))

            if draw_overlap:
                rollout_players_str = \
                    draw_all_players(cur_rollout_row, cur_rollout_pred_row, rollout_d2_img_draw, draw_max,
                                     rollout_players_to_draw, draw_only_pos=True, player_to_color=rollout_colors)
                manual_players_str = \
                    draw_all_players(cur_manual_row, cur_manual_pred_row, manual_d2_img_draw, draw_max,
                                     manual_players_to_draw, draw_only_pos=True, player_to_color=manual_colors,
                                     rectangle=False)
                draw_player_connection_lines(cur_rollout_row, cur_manual_row, manual_d2_img_draw,
                                             rollout_to_manual_round_data.agent_mapping, rollout_players_to_draw,
                                             rollout_colors)
            else:
                rollout_players_str = \
                    draw_all_players(cur_rollout_row, cur_rollout_pred_row, rollout_d2_img_draw, draw_max,
                                     rollout_players_to_draw)
                manual_players_str = \
                    draw_all_players(cur_manual_row, cur_manual_pred_row, manual_d2_img_draw, draw_max,
                                     manual_players_to_draw)
            details_text_var.set("rollout\n" + rollout_players_str + "\nmanual\n" + manual_players_str)

            if draw_overlap:
                rollout_d2_img_copy.alpha_composite(rollout_d2_overlay_im)
                rollout_d2_img_copy.alpha_composite(manual_d2_overlay_im)
                updated_rollout_d2_photo_img = itk.PhotoImage(rollout_d2_img_copy)
                rollout_img_label.configure(image=updated_rollout_d2_photo_img)
                rollout_img_label.image = updated_rollout_d2_photo_img
            else:
                rollout_d2_img_copy.alpha_composite(rollout_d2_overlay_im)
                updated_rollout_d2_photo_img = itk.PhotoImage(rollout_d2_img_copy)
                rollout_img_label.configure(image=updated_rollout_d2_photo_img)
                rollout_img_label.image = updated_rollout_d2_photo_img

                manual_d2_img_copy.alpha_composite(manual_d2_overlay_im)
                updated_manual_d2_photo_img = itk.PhotoImage(manual_d2_img_copy)
                manual_img_label.configure(image=updated_manual_d2_photo_img)
                manual_img_label.image = updated_manual_d2_photo_img

    def step_back_clicked():
        nonlocal cur_similarity_tick_index
        if cur_similarity_tick_index > 0:
            cur_similarity_tick_index -= 1
            tick_slider.set(cur_similarity_tick_index)
            tick_slider_changed(cur_similarity_tick_index)


    play_active: bool = False
    num_play_updates_sleeping: int = 0
    def play_clicked():
        nonlocal play_active, num_play_updates_sleeping
        play_active = not play_active
        if play_active:
            play_button.configure(bg='green')
            # technically not sleeping, but need to increment by 1 so -= 1 math works out
            num_play_updates_sleeping += 1
            play_update()
        else:
            play_button.configure(bg=orig_player_button_color)


    def play_update():
        nonlocal num_play_updates_sleeping
        num_play_updates_sleeping -= 1
        if play_active and num_play_updates_sleeping == 0:
            step_forward_clicked()
            num_play_updates_sleeping += 1
            play_button.after(250, play_update)


    def step_forward_clicked():
        nonlocal cur_similarity_tick_index
        if cur_similarity_tick_index < len(similarity_match_df) - 1:
            cur_similarity_tick_index += 1
            tick_slider.set(cur_similarity_tick_index)
            tick_slider_changed(cur_similarity_tick_index)


    def toggle_distribution_clicked():
        nonlocal draw_max
        draw_max = not draw_max
        tick_slider_changed(cur_similarity_tick_index)


    def toggle_overlap_clicked():
        nonlocal draw_overlap
        draw_overlap = not draw_overlap
        if draw_overlap:
            manual_img_label.pack_forget()
        else:
            manual_img_label.pack(side="left")
        tick_slider_changed(cur_similarity_tick_index)

    # state setters
    def change_round_metric_dependent_data():
        nonlocal rollout_selected_df, rollout_pred_selected_df, manual_selected_df, manual_pred_selected_df, \
            similarity_match_df, cur_round, cur_metric_type, rollout_to_manual_round_data
        rollout_selected_df = rollout_data_df.loc[rollout_data_df[round_id_column] == cur_round]
        rollout_pred_selected_df = rollout_pred_df.loc[rollout_data_df[round_id_column] == cur_round]

        rollout_to_manual_round_data = rollout_to_manual_dict[cur_round][cur_metric_type]
        manual_selected_df = \
            manual_data_df.loc[manual_data_df[round_id_column] == rollout_to_manual_round_data.manual_round_id]
        manual_pred_selected_df = \
            manual_pred_df.loc[manual_data_df[round_id_column] == rollout_to_manual_round_data.manual_round_id]
        similarity_match_df = rollout_to_manual_round_data.similarity_match_df

        tick_slider.configure(to=len(similarity_match_df)-1)
        tick_slider.set(0)
        tick_slider_changed(0)


    s = ttk.Style()
    s.theme_use('alt')
    s.configure('Valid.TCombobox', fieldbackground='white')
    s.configure('Invalid.TCombobox', fieldbackground='#cfcfcf')
    # creating round slider and label
    round_id_frame = tk.Frame(window)
    round_id_frame.pack(pady=5)
    round_id_text_var = tk.StringVar()
    round_id_label = tk.Label(round_id_frame, textvariable=round_id_text_var)
    round_id_label.pack(side="left")

    round_frame = tk.Frame(window)
    round_frame.pack(pady=5)

    round_slider = tk.Scale(
        round_frame,
        from_=0,
        to=len(rounds)-1,
        orient='horizontal',
        showvalue=0,
        length=500,
        command=round_slider_changed
    )
    round_slider.pack(side="left")

    # round id stepper
    back_round_button = tk.Button(round_frame, text="<<", command=round_back_clicked)
    back_round_button.pack(side="left")
    forward_round_button = tk.Button(round_frame, text=">>", command=round_forward_clicked)
    forward_round_button.pack(side="left")

    # creating metric slider and label
    metric_id_frame = tk.Frame(window)
    metric_id_frame.pack(pady=5)
    metric_id_text_var = tk.StringVar()
    metric_id_label = tk.Label(metric_id_frame, textvariable=metric_id_text_var)
    metric_id_label.pack(side="left")

    metric_frame = tk.Frame(window)
    metric_frame.pack(pady=5)

    metric_slider = tk.Scale(
        metric_frame,
        from_=0,
        to=len(metric_types)-1,
        orient='horizontal',
        showvalue=0,
        length=500,
        command=metric_slider_changed
    )
    metric_slider.pack(side="left")

    # round id stepper
    back_metric_button = tk.Button(metric_frame, text="<<", command=metric_back_clicked)
    back_metric_button.pack(side="left")
    forward_metric_button = tk.Button(metric_frame, text=">>", command=metric_forward_clicked)
    forward_metric_button.pack(side="left")

    # creating tick slider and label
    tick_id_frame = tk.Frame(window)
    tick_id_frame.pack(pady=5)

    tick_id_text_var = tk.StringVar()
    tick_id_label = tk.Label(tick_id_frame, textvariable=tick_id_text_var)
    tick_id_label.pack(side="left")

    tick_slider_frame = tk.Frame(window)
    tick_slider_frame.pack(pady=5)

    tick_slider = tk.Scale(
        tick_slider_frame,
        from_=0,
        to=100,
        orient='horizontal',
        showvalue=0,
        length=500,
        command=tick_slider_changed
    )
    tick_slider.pack(side="left")

    # creating tick play/pause/step buttons
    back_step_button = tk.Button(tick_slider_frame, text="<<", command=step_back_clicked)
    back_step_button.pack(side="left")
    play_button = tk.Button(tick_slider_frame, text="|>", command=play_clicked)
    orig_player_button_color = play_button.cget("background")
    play_button.pack(side="left")
    forward_step_button = tk.Button(tick_slider_frame, text=">>", command=step_forward_clicked)
    forward_step_button.pack(side="left")

    # creating vis control frame
    distribution_control_frame = tk.Frame(window)
    distribution_control_frame.pack(pady=5)
    distribution_toggle_button = tk.Button(distribution_control_frame, text="toggle max/distribution", command=toggle_distribution_clicked)
    distribution_toggle_button.pack(side="left")
    player_distributions_label = tk.Label(distribution_control_frame, text="players to show")
    player_distributions_label.pack(side="left")
    player_distributions_var = tk.StringVar(value="*")
    player_distributions_entry = tk.Entry(distribution_control_frame, width=30, textvariable=player_distributions_var)
    player_distributions_entry.pack(side="left")
    overlap_toggle_button = tk.Button(distribution_control_frame, text="toggle overlap", command=toggle_overlap_clicked)
    overlap_toggle_button.pack(side="left")


    details_frame = tk.Frame(window)
    details_frame.pack(pady=5)

    details_text_var = tk.StringVar()
    details_label = tk.Label(details_frame, textvariable=details_text_var, anchor="e", justify=tk.LEFT)
    details_label.pack(side="left")

    # initial value settings
    round_slider_changed(0)
    metric_slider_changed(0)
    tick_slider_changed(0)

    # Start the GUI
    window.mainloop()
