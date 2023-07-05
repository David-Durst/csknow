from typing import Dict

import pandas as pd

from learn_bot.latent.load_model import LoadedModel
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
class PredictedToGroundTruthRoundData:
    predicted_round_id: int
    ground_truth_round_id: int
    ground_truth_hdf5_filename: str
    similarity_row: pd.Series
    agent_mapping: Dict[int, int]

    def get_similarity_match_index_df_subset(self, similarity_match_index_df: pd.DataFrame) -> pd.DataFrame:
        return similarity_match_index_df.iloc[
            self.similarity_row[start_dtw_matched_indices_col]:
            self.similarity_row[start_dtw_matched_indices_col] + self.similarity_row[length_dtw_matched_inidices_col]]


PredictedToGroundTruthDict = Dict[str, Dict[int, Dict[str, List[PredictedToGroundTruthRoundData]]]]

cmap = mpl.cm.get_cmap("Set3").colors


def vis_two(predicted_model: LoadedModel, ground_truth_model: LoadedModel,
            predicted_to_ground_truth_dict: PredictedToGroundTruthDict, similarity_match_index_df: pd.DataFrame):
    make_index_column(predicted_model.cur_loaded_df)
    make_index_column(ground_truth_model.cur_loaded_df)

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
    predicted_d2_img_draw = d2_img.copy()
    predicted_d2_photo_img = itk.PhotoImage(predicted_d2_img_draw)
    predicted_img_label = tk.Label(img_frame, image=predicted_d2_photo_img)
    predicted_img_label.pack(side="left")
    ground_truth_d2_img_draw = d2_img.copy()
    ground_truth_d2_photo_img = itk.PhotoImage(ground_truth_d2_img_draw)
    ground_truth_img_label = tk.Label(img_frame, image=ground_truth_d2_photo_img)
    ground_truth_img_label.pack(side="left")

    # can't use unique round ids because some rounds may have no matches (on human data, with weird num alive counts)
    rounds = list(predicted_to_ground_truth_dict[predicted_model.get_cur_hdf5_filename()].keys())
    cur_round: int = -1
    metric_types = ['Unconstrained DTW', 'Slope Constrained DTW', 'Percentile ADE']
    cur_metric_type: str = metric_types[0]
    cur_round_match_id: int = -1
    num_round_matches: int = 5
    cur_similarity_tick_index: int = -1
    predicted_selected_df: pd.DataFrame = predicted_model.cur_loaded_df
    ground_truth_selected_df: pd.DataFrame = ground_truth_model.cur_loaded_df
    similarity_match_index_subset_df: Optional[pd.DataFrame] = None
    predicted_to_ground_truth_round_data: Optional[PredictedToGroundTruthRoundData] = None
    draw_max: bool = True
    draw_overlap: bool = False

    def hdf5_id_update():
        nonlocal rounds, cur_round
        predicted_model.cur_hdf5_index = int(new_hdf5_id_entry.get())
        predicted_model.load_cur_hdf5_as_pd()
        make_index_column(predicted_model.cur_loaded_df)
        rounds = list(predicted_to_ground_truth_dict[predicted_model.get_cur_hdf5_filename()].keys())
        cur_round = rounds[0]
        round_slider.set(0)
        round_slider_changed(0)

    def round_slider_changed(cur_round_index):
        nonlocal cur_round, num_round_matches
        cur_round = rounds[int(cur_round_index)]
        #if predicted_model.get_cur_hdf5_filename() not in predicted_to_ground_truth_dict:
        #    print(1)
        #if cur_round not in predicted_to_ground_truth_dict[predicted_model.get_cur_hdf5_filename()]:
        #    print(2)
        #if cur_metric_type not in predicted_to_ground_truth_dict[predicted_model.get_cur_hdf5_filename()][cur_round]:
        #    print(3)
        num_round_matches = len(predicted_to_ground_truth_dict[predicted_model.get_cur_hdf5_filename()][cur_round][cur_metric_type])
        round_match_slider.configure(to=num_round_matches - 1)
        round_match_slider_changed(0)

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

    def round_match_slider_changed(cur_round_match_index):
        nonlocal cur_round_match_id
        cur_round_match_id = int(cur_round_match_index)
        change_round_metric_dependent_data()

    def round_match_back_clicked():
        nonlocal cur_round_match_id
        cur_round_match_id = int(round_match_slider.get())
        if cur_round_match_id > 0:
            cur_round_match_id -= 1
            round_match_slider.set(cur_round_match_id)
            round_match_slider_changed(cur_round_match_id)

    def round_match_forward_clicked():
        nonlocal cur_round_match_id
        cur_round_match_id = int(round_match_slider.get())
        if cur_round_match_id < num_round_matches - 1:
            cur_round_match_id += 1
            round_match_slider.set(cur_round_match_id)
            round_match_slider_changed(cur_round_match_id)

    predicted_players_active = []
    ground_truth_players_active = []
    player_index_mapping = {}
    ground_truth_players_to_draw = []

    def tick_slider_changed(cur_similarity_tick_index_str):
        nonlocal cur_similarity_tick_index, d2_img, predicted_d2_img_draw, predicted_img_label, \
            ground_truth_d2_img_draw, ground_truth_img_label, draw_max, draw_overlap, \
            predicted_players_active, ground_truth_players_active, player_index_mapping, ground_truth_players_to_draw
        if len(predicted_selected_df) > 0:
            cur_similarity_tick_index = int(cur_similarity_tick_index_str)

            cur_predicted_index = similarity_match_index_subset_df.iloc[cur_similarity_tick_index].loc[first_matched_index_col]
            cur_predicted_row = predicted_selected_df.iloc[cur_predicted_index]
            cur_predicted_tick_id = cur_predicted_row.loc[tick_id_column]
            cur_predicted_game_tick_id = cur_predicted_row.loc[tick_id_column]

            cur_ground_truth_index = similarity_match_index_subset_df.iloc[cur_similarity_tick_index].loc[second_matched_index_col]
            cur_ground_truth_row = ground_truth_selected_df.iloc[cur_ground_truth_index]
            cur_ground_truth_round = cur_ground_truth_row.loc[round_id_column]
            cur_ground_truth_tick_id = cur_ground_truth_row.loc[tick_id_column]
            cur_ground_truth_game_tick_id = cur_ground_truth_row.loc[game_tick_number_column]

            hdf5_id_text_var.set(f"Predicted Cur HDF5 Id: {predicted_model.get_cur_hdf5_filename()} - {predicted_model.cur_hdf5_index} / {len(predicted_model.dataset.data_hdf5s)}, "
                                 f"Ground Truth Cur HDF5 Id: {ground_truth_model.get_cur_hdf5_filename()} - {ground_truth_model.cur_hdf5_index} / {len(ground_truth_model.dataset.data_hdf5s)}, ")
            tick_id_text_var.set(f"Predicted Tick ID: {cur_predicted_tick_id}, Predicted Game Tick ID: {cur_predicted_game_tick_id}, "
                                 f"Ground Truth Tick Id: {cur_ground_truth_tick_id}, Ground Truth Game Tick ID: {cur_ground_truth_game_tick_id}")
            round_id_text_var.set(f"Predicted Round ID: {int(cur_round)}, Predicted Round Number: {cur_predicted_row.loc['round number']}, "
                                  f"Ground Truth Round ID: {int(cur_ground_truth_round)}, Ground Truth Round Number: {cur_ground_truth_row.loc['round number']}")
            metric_id_text_var.set(cur_metric_type)
            round_match_id_text_var.set(f"Round Match {cur_round_match_id} / {num_round_matches - 1}, "
                                        f"DTW/ADE Cost {predicted_to_ground_truth_round_data.similarity_row[dtw_cost_col]: .2f}, "
                                        f"Delta Time {predicted_to_ground_truth_round_data.similarity_row[delta_time_col]: .2f}, "
                                        f"Delta Distance {predicted_to_ground_truth_round_data.similarity_row[delta_distance_col]: .2f}")

            predicted_d2_img_copy = d2_img.copy().convert("RGBA")
            predicted_d2_overlay_im = Image.new("RGBA", predicted_d2_img_copy.size, (255, 255, 255, 0))
            predicted_d2_img_draw = ImageDraw.Draw(predicted_d2_overlay_im)
            ground_truth_d2_img_copy = d2_img.copy().convert("RGBA")
            ground_truth_d2_overlay_im = Image.new("RGBA", ground_truth_d2_img_copy.size, (255, 255, 255, 0))
            ground_truth_d2_img_draw = ImageDraw.Draw(ground_truth_d2_overlay_im)

            players_to_draw_str = player_distributions_var.get()
            if players_to_draw_str == "*":
                predicted_players_to_draw = list(range(0, len(specific_player_place_area_columns)))
            else:
                predicted_players_to_draw = [int(p) for p in players_to_draw_str.split(";")[0].split(",")]

            # need to update these so only change once per round, at start when all players alive
            # converting agent mapping (which is monotonically increasing from 0) to column index, which has
            # dead players interspersed
            if cur_similarity_tick_index == 0:
                predicted_players_active = []
                ground_truth_players_active = []
                for player_index in range(len(specific_player_place_area_columns)):
                    if cur_predicted_row[specific_player_place_area_columns[player_index].player_id] != -1:
                        predicted_players_active.append(player_index)
                    if cur_ground_truth_row[specific_player_place_area_columns[player_index].player_id] != -1:
                        ground_truth_players_active.append(player_index)
                player_index_mapping = {}
                for src, tgt in predicted_to_ground_truth_round_data.agent_mapping.items():
                    player_index_mapping[predicted_players_active[src]] = ground_truth_players_active[tgt]
                ground_truth_players_to_draw = []
                for p in predicted_players_to_draw:
                    if p in player_index_mapping:
                        ground_truth_players_to_draw.append(player_index_mapping[p])

            predicted_colors = {}
            ground_truth_colors = {}
            for predicted_agent_num, ground_truth_agent_num in predicted_to_ground_truth_round_data.agent_mapping.items():
                predicted_column_index = predicted_players_active[predicted_agent_num]
                ground_truth_column_index = ground_truth_players_active[ground_truth_agent_num]
                predicted_color = cmap[predicted_column_index]
                predicted_colors[predicted_column_index] = (int(255 * predicted_color[0]), int(255 * predicted_color[1]),
                                                            int(255 * predicted_color[2]))
                ground_truth_color = cmap[predicted_column_index]
                ground_truth_colors[ground_truth_column_index] = (int(255 * ground_truth_color[0]),
                                                                  int(255 * ground_truth_color[1]),
                                                                  int(255 * ground_truth_color[2]))

            if draw_overlap:
                predicted_players_str = \
                    draw_all_players(cur_predicted_row, None, predicted_d2_img_draw, draw_max,
                                     predicted_players_to_draw, draw_only_pos=True, player_to_color=predicted_colors)
                ground_truth_players_str = \
                    draw_all_players(cur_ground_truth_row, None, ground_truth_d2_img_draw, draw_max,
                                     ground_truth_players_to_draw, draw_only_pos=True, player_to_color=ground_truth_colors,
                                     rectangle=False)
                draw_player_connection_lines(cur_predicted_row, cur_ground_truth_row, ground_truth_d2_img_draw,
                                             player_index_mapping, predicted_players_to_draw, predicted_colors)
            else:
                predicted_players_str = \
                    draw_all_players(cur_predicted_row, None, predicted_d2_img_draw, draw_max,
                                     predicted_players_to_draw, draw_only_pos=True)
                ground_truth_players_str = \
                    draw_all_players(cur_ground_truth_row, None, ground_truth_d2_img_draw, draw_max,
                                     ground_truth_players_to_draw, draw_only_pos=True)
            details_text_var.set("predicted\n" + predicted_players_str + "\nground_truth\n" + ground_truth_players_str)

            if draw_overlap:
                predicted_d2_img_copy.alpha_composite(predicted_d2_overlay_im)
                predicted_d2_img_copy.alpha_composite(ground_truth_d2_overlay_im)
                updated_predicted_d2_photo_img = itk.PhotoImage(predicted_d2_img_copy)
                predicted_img_label.configure(image=updated_predicted_d2_photo_img)
                predicted_img_label.image = updated_predicted_d2_photo_img
            else:
                predicted_d2_img_copy.alpha_composite(predicted_d2_overlay_im)
                updated_predicted_d2_photo_img = itk.PhotoImage(predicted_d2_img_copy)
                predicted_img_label.configure(image=updated_predicted_d2_photo_img)
                predicted_img_label.image = updated_predicted_d2_photo_img

                ground_truth_d2_img_copy.alpha_composite(ground_truth_d2_overlay_im)
                updated_ground_truth_d2_photo_img = itk.PhotoImage(ground_truth_d2_img_copy)
                ground_truth_img_label.configure(image=updated_ground_truth_d2_photo_img)
                ground_truth_img_label.image = updated_ground_truth_d2_photo_img

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
        if cur_similarity_tick_index < len(similarity_match_index_subset_df) - 1:
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
            ground_truth_img_label.pack_forget()
        else:
            ground_truth_img_label.pack(side="left")
        tick_slider_changed(cur_similarity_tick_index)

    # state setters
    def change_round_metric_dependent_data():
        nonlocal predicted_selected_df, ground_truth_selected_df, similarity_match_index_subset_df, cur_round, cur_metric_type, \
            predicted_to_ground_truth_round_data
        predicted_selected_df = \
            predicted_model.cur_loaded_df.loc[predicted_model.cur_loaded_df[round_id_column] == cur_round]

        predicted_to_ground_truth_round_data = \
            predicted_to_ground_truth_dict[predicted_model.get_cur_hdf5_filename()][cur_round][cur_metric_type][cur_round_match_id]

        new_ground_truth_hdf5_filename = predicted_to_ground_truth_round_data.ground_truth_hdf5_filename
        new_ground_truth_hdf5_index = ground_truth_model.filename_to_hdf5_index[new_ground_truth_hdf5_filename]
        if new_ground_truth_hdf5_index != ground_truth_model.cur_hdf5_index:
            ground_truth_model.cur_hdf5_index = new_ground_truth_hdf5_index
            ground_truth_model.load_cur_hdf5_as_pd()
            make_index_column(ground_truth_model.cur_loaded_df)
        ground_truth_selected_df = \
            ground_truth_model.cur_loaded_df.loc[ground_truth_model.cur_loaded_df[round_id_column] ==
                                                 predicted_to_ground_truth_round_data.ground_truth_round_id]
        similarity_match_index_subset_df = \
            predicted_to_ground_truth_round_data.get_similarity_match_index_df_subset(similarity_match_index_df)

        tick_slider.configure(to=len(similarity_match_index_subset_df)-1)
        tick_slider.set(0)
        tick_slider_changed(0)


    s = ttk.Style()
    s.theme_use('alt')
    s.configure('Valid.TCombobox', fieldbackground='white')
    s.configure('Invalid.TCombobox', fieldbackground='#cfcfcf')
    # hdf5 id slider
    hdf5_id_frame = tk.Frame(window)
    hdf5_id_frame.pack(pady=5)
    hdf5_id_text_var = tk.StringVar()
    hdf5_id_label = tk.Label(hdf5_id_frame, textvariable=hdf5_id_text_var)
    hdf5_id_label.pack(side="left")
    new_hdf5_id_label = tk.Label(hdf5_id_frame, text="New HDF5 Id:")
    new_hdf5_id_label.pack(side="left")
    new_hdf5_id_var = tk.StringVar(value="")
    new_hdf5_id_entry = tk.Entry(hdf5_id_frame, width=5, textvariable=new_hdf5_id_var)
    new_hdf5_id_entry.pack(side="left")
    update_hdf5_id_button = tk.Button(hdf5_id_frame, text="update hdf5 id", command=hdf5_id_update)
    update_hdf5_id_button.pack(side="left")

    # round id slider
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

    # metric id stepper
    back_metric_button = tk.Button(metric_frame, text="<<", command=metric_back_clicked)
    back_metric_button.pack(side="left")
    forward_metric_button = tk.Button(metric_frame, text=">>", command=metric_forward_clicked)
    forward_metric_button.pack(side="left")

    # creating round_match number slider and label
    round_match_id_frame = tk.Frame(window)
    round_match_id_frame.pack(pady=5)
    round_match_id_text_var = tk.StringVar()
    round_match_id_label = tk.Label(round_match_id_frame, textvariable=round_match_id_text_var)
    round_match_id_label.pack(side="left")

    round_match_frame = tk.Frame(window)
    round_match_frame.pack(pady=5)

    round_match_slider = tk.Scale(
        round_match_frame,
        from_=0,
        to=10,
        orient='horizontal',
        showvalue=0,
        length=500,
        command=round_match_slider_changed
    )
    round_match_slider.pack(side="left")

    # round_match id stepper
    back_round_match_button = tk.Button(round_match_frame, text="<<", command=round_match_back_clicked)
    back_round_match_button.pack(side="left")
    forward_round_match_button = tk.Button(round_match_frame, text=">>", command=round_match_forward_clicked)
    forward_round_match_button.pack(side="left")

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
