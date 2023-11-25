from typing import Set, Callable, List

from learn_bot.latent.place_area.push_save_label import PushSaveRoundLabels, PushSaveRoundData, PushSaveLabel
from learn_bot.libs.pd_printing import set_pd_print_options
from learn_bot.latent.place_area.column_names import *
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.train_paths import default_selected_retake_rounds_path, default_save_push_round_labels_path
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.mining.area_cluster import d2_radar_path
from learn_bot.latent.vis.draw_inference import draw_all_players, minimap_height, minimap_width, scale_down
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageDraw, ImageTk as itk
import pandas as pd


def get_rounds_for_cur_hdf5(loaded_model: LoadedModel) -> List[int]:
    return loaded_model.get_cur_id_df().loc[:, round_id_column].unique().tolist()


def index_cur_hdf5(loaded_model: LoadedModel, index_inference_df: bool = True):
    #make_index_column(loaded_model.cur_loaded_df)
    if index_inference_df:
        make_index_column(loaded_model.cur_inference_df)
    make_index_column(loaded_model.get_cur_id_df())


ct_color = (4, 190, 196)
t_color = (187, 142, 52)


def vis(loaded_model: LoadedModel, inference_fn: Callable[[LoadedModel], None], window_title_appendix: str = "",
        use_sim_dataset: bool = False):
    inference_fn(loaded_model)
    index_cur_hdf5(loaded_model)

    #This creates the main window of an application
    window = tk.Tk()
    window.title("Delta Position Model " + window_title_appendix)
    window.resizable(width=True, height=True)
    window.configure(background='grey')

    your_font = font.nametofont("TkDefaultFont")  # Get default font value into Font object
    your_font.actual()

    img_frame = tk.Frame(window)
    img_frame.pack(pady=5)
    d2_img = Image.open(d2_radar_path)
    scale_down()
    d2_img = d2_img.resize((minimap_width(), minimap_height()), Image.ANTIALIAS)
    d2_img_draw = d2_img.copy()
    d2_photo_img = itk.PhotoImage(d2_img_draw)
    img_label = tk.Label(img_frame, image=d2_photo_img)
    img_label.pack(side="left")

    rounds = get_rounds_for_cur_hdf5(loaded_model)
    print(f"num rounds {len(rounds)}")
    ticks = []
    game_ticks = []
    cur_round_id: int = -1
    cur_round_number: int = -1
    cur_tick: int = -1
    cur_tick_index: int = -1
    selected_df = loaded_model.load_round_df_from_cur_dataset(cur_round_id, use_sim_dataset=use_sim_dataset)
    pred_selected_df: pd.DataFrame = loaded_model.cur_inference_df
    id_df: pd.DataFrame = loaded_model.get_cur_id_df()
    draw_pred: bool = True
    draw_max: bool = True
    push_save_round_labels: PushSaveRoundLabels = PushSaveRoundLabels()

    set_pd_print_options()

    def hdf5_id_update():
        nonlocal rounds, cur_round_id
        loaded_model.cur_hdf5_index = int(new_hdf5_id_entry.get())
        loaded_model.load_cur_dataset_only()
        inference_fn(loaded_model)
        index_cur_hdf5(loaded_model)
        rounds = get_rounds_for_cur_hdf5(loaded_model)
        cur_round_id = rounds[0]
        print(f"num rounds {len(rounds)}")
        round_slider.set(0)
        round_slider.configure(to=len(rounds) - 1)
        round_slider_changed(0)

    def round_slider_changed(cur_round_index):
        nonlocal cur_round_id
        cur_round_id = rounds[int(cur_round_index)]
        change_round_dependent_data()

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

    def tick_slider_changed(cur_tick_index_str):
        nonlocal cur_tick, cur_tick_index, cur_round_number, d2_img, d2_img_draw, img_label, draw_max
        if selected_df.empty:
            print("why empty selected_df?")
            return
        cur_tick_index = int(cur_tick_index_str)
        cur_tick = ticks[cur_tick_index]
        cur_game_tick = game_ticks[cur_tick_index]

        hdf5_id_text_var.set(f"Predicted Cur HDF5 Id: {loaded_model.get_cur_hdf5_filename()} - {loaded_model.cur_hdf5_index} / "
                             f"{len(loaded_model.dataset.data_hdf5s) - 1}, ")
        tick_id_text_var.set("Tick ID: " + str(cur_tick))
        data_dict = selected_df.iloc[[cur_tick_index], :].to_dict('records')[0]
        pred_dict = pred_selected_df.iloc[[cur_tick_index], :].to_dict('records')[0]
        tick_game_id_text_var.set(f"Game Tick ID: {cur_game_tick}")
        extra_round_data_str = ""
        if get_similarity_column(0) in id_df.columns:
            extra_round_data_str = f"similarity 0: {id_df.iloc[cur_tick_index].loc[get_similarity_column(0)]}"
        game_id = data_dict[game_id_column]
        demo_file_text_var.set(loaded_model.cur_demo_names[game_id])
        round_id_text_var.set(f"Round ID: {int(cur_round_id)}, "
                              f"Round Number: {data_dict[round_number_column]}, {extra_round_data_str}")
        cur_round_number = data_dict[round_number_column]
        other_state_text_var.set(f"Planted A {data_dict[c4_plant_a_col]}, "
                                 f"Planted B {data_dict[c4_plant_b_col]}, "
                                 f"Not Planted {data_dict[c4_not_planted_col]}, "
                                 f"C4 Pos ({data_dict[c4_pos_cols[0]]:.2f}, {data_dict[c4_pos_cols[1]]:.2f}, {data_dict[c4_pos_cols[2]]:.2f}),"
                                 f"C4 Time Left Percent {data_dict[c4_time_left_percent[0]]:.2f}")

        d2_img_copy = d2_img.copy().convert("RGBA")
        d2_overlay_im = Image.new("RGBA", d2_img_copy.size, (255, 255, 255, 0))
        d2_img_draw = ImageDraw.Draw(d2_overlay_im)
        players_to_draw_str = player_distributions_var.get()
        if players_to_draw_str == "*":
            players_to_draw = list(range(0, len(specific_player_place_area_columns)))
        else:
            players_to_draw = [int(p) for p in players_to_draw_str.split(",")]
        player_to_color = {}
        if not draw_pred:
            for i in range(max_enemies):
                player_to_color[i] = ct_color
                player_to_color[i+max_enemies] = t_color
        players_status = draw_all_players(data_dict, pred_dict, d2_img_draw, draw_max, players_to_draw,
                                          player_to_color=player_to_color, draw_only_pos=not draw_pred,
                                          radial_vel_time_step=radial_vel_time_step_slider.get())

        details_text_var.set(players_status.status + "\n\n" + players_status.temporal)
        d2_img_copy.alpha_composite(d2_overlay_im)
        updated_d2_photo_img = itk.PhotoImage(d2_img_copy)
        img_label.configure(image=updated_d2_photo_img)
        img_label.image = updated_d2_photo_img

    def step_back_clicked():
        nonlocal cur_tick_index
        if cur_tick_index > 0:
            cur_tick_index -= 1
            tick_slider.set(cur_tick_index)
            tick_slider_changed(cur_tick_index)


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
            play_button.after(63, play_update)


    def step_forward_clicked():
        nonlocal cur_tick_index
        if cur_tick_index < len(ticks) - 1:
            cur_tick_index += 1
            tick_slider.set(cur_tick_index)
            tick_slider_changed(cur_tick_index)

    def pred_toggle_clicked():
        nonlocal draw_pred
        draw_pred = not draw_pred
        tick_slider_changed(cur_tick_index)

    def print_row_clicked():
        non_delta_pos_cols = []
        for col in selected_df.columns:
            if "delta pos" not in col:
                non_delta_pos_cols.append(col)
        data_series = selected_df.iloc[cur_tick_index].loc[non_delta_pos_cols]
        print(data_series)

    def toggle_distribution_clicked():
        nonlocal draw_max
        draw_max = not draw_max
        tick_slider_changed(cur_tick_index)

    def radial_vel_time_step_slider_changed(new_radial_vel_time_step):
        tick_slider_changed(cur_tick_index)

    def load_push_save_labels():
        push_save_round_labels.load(selected_retake_rounds_path_var.get())
        if cur_round_id in push_save_round_labels.round_id_to_data:
            push_save_label_var.set(int(push_save_round_labels.round_id_to_data[cur_round_id].label))
        else:
            push_save_label_var.set(0)

    def push_save_label_clicked():
        if push_save_label_var.get() == 0:
            push_save_round_labels.round_id_to_data.pop(cur_round_id, None)
        else:
            push_save_round_labels.round_id_to_data[cur_round_id] = PushSaveRoundData(
                PushSaveLabel(push_save_label_var.get()),
                tick_slider.get() / len(ticks),
                demo_file_text_var.get(),
                cur_round_id,
                cur_round_number
            )
        push_save_round_labels.save(selected_retake_rounds_path_var.get())

    # state setters
    def change_round_dependent_data():
        nonlocal selected_df, id_df, pred_selected_df, cur_round_id, ticks, game_ticks
        vis_only_df = loaded_model.get_cur_vis_df()
        make_index_column(vis_only_df)
        selected_df = loaded_model.load_round_df_from_cur_dataset(cur_round_id, vis_only_df,
                                                                  use_sim_dataset=use_sim_dataset)
        cur_round_condition = loaded_model.get_cur_id_df()[round_id_column] == cur_round_id
        id_df = loaded_model.get_cur_id_df().loc[cur_round_condition]
        pred_selected_df = loaded_model.cur_inference_df.loc[cur_round_condition]
        pred_selected_df = pred_selected_df.reset_index(drop=True)

        ticks = selected_df.loc[:, 'tick id'].tolist()
        game_ticks = selected_df.loc[:, 'game tick number'].tolist()
        tick_slider.configure(to=len(ticks)-1)
        tick_slider.set(0)
        radial_vel_time_step_slider.set(0)
        tick_slider_changed(0)
        if cur_round_id in push_save_round_labels.round_id_to_data:
            push_save_label_var.set(int(push_save_round_labels.round_id_to_data[cur_round_id].label))
        else:
            push_save_label_var.set(0)


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
    demo_file_label = tk.Label(round_id_frame, text="Demo File:")
    demo_file_label.pack(side="left")
    demo_file_text_var = tk.StringVar()
    demo_file_entry = tk.Entry(round_id_frame, width=50, textvariable=demo_file_text_var)
    demo_file_entry.pack(side="left")
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
        showvalue=False,
        length=500,
        command=round_slider_changed
    )
    round_slider.pack(side="left")

    # round id stepper
    back_round_button = tk.Button(round_frame, text="<<", command=round_back_clicked)
    back_round_button.pack(side="left")
    forward_round_button = tk.Button(round_frame, text=">>", command=round_forward_clicked)
    forward_round_button.pack(side="left")

    # creating tick slider and label
    tick_id_frame = tk.Frame(window)
    tick_id_frame.pack(pady=5)

    tick_id_text_var = tk.StringVar()
    tick_id_label = tk.Label(tick_id_frame, textvariable=tick_id_text_var)
    tick_id_label.pack(side="left")

    tick_demo_id_text_var = tk.StringVar()
    tick_demo_id_label = tk.Label(tick_id_frame, textvariable=tick_demo_id_text_var)
    tick_demo_id_label.pack(side="left")

    tick_game_id_text_var = tk.StringVar()
    tick_game_id_label = tk.Label(tick_id_frame, textvariable=tick_game_id_text_var)
    tick_game_id_label.pack(side="left")

    tick_slider_frame = tk.Frame(window)
    tick_slider_frame.pack(pady=5)

    tick_slider = tk.Scale(
        tick_slider_frame,
        from_=0,
        to=100,
        orient='horizontal',
        showvalue=False,
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
    pred_toggle_button = tk.Button(distribution_control_frame, text="toggle pred", command=pred_toggle_clicked)
    pred_toggle_button.pack(side="left")
    print_row_button = tk.Button(distribution_control_frame, text="print df row", command=print_row_clicked)
    print_row_button.pack(side="left")
    distribution_toggle_button = tk.Button(distribution_control_frame, text="toggle max/distribution", command=toggle_distribution_clicked)
    distribution_toggle_button.pack(side="left")
    player_distributions_label = tk.Label(distribution_control_frame, text="players to show")
    player_distributions_label.pack(side="left")
    player_distributions_var = tk.StringVar(value="*")
    player_distributions_entry = tk.Entry(distribution_control_frame, width=30, textvariable=player_distributions_var)
    player_distributions_entry.pack(side="left")
    radial_vel_time_step_label = tk.Label(distribution_control_frame, text="Radial Vel Time Step: ")
    radial_vel_time_step_label.pack(side="left")
    radial_vel_time_step_slider = tk.Scale(
        distribution_control_frame,
        from_=0,
        to=2,
        orient='horizontal',
        showvalue=True,
        length=120,
        command=radial_vel_time_step_slider_changed
    )
    radial_vel_time_step_slider.pack(side="left")

    round_label_frame = tk.Frame(window)
    round_label_frame.pack(pady=5)
    selected_retake_rounds_path_var = tk.StringVar(value=default_save_push_round_labels_path)
    selected_retake_rounds_entry = tk.Entry(round_label_frame, width=50, textvariable=selected_retake_rounds_path_var)
    selected_retake_rounds_entry.pack(side="left")
    load_selected_retake_rounds_button = tk.Button(round_label_frame, text="load push/save labels", command=load_push_save_labels)
    load_selected_retake_rounds_button.pack(side="left")
    push_save_label_var = tk.IntVar(value=0)
    unlabeled_round_radio = tk.Radiobutton(round_label_frame, text="unlabeled", variable=push_save_label_var,
                                           value=0, command=push_save_label_clicked)
    unlabeled_round_radio.pack(side="left")
    push_round_radio = tk.Radiobutton(round_label_frame, text="push", variable=push_save_label_var,
                                           value=1, command=push_save_label_clicked)
    push_round_radio.pack(side="left")
    partial_round_radio = tk.Radiobutton(round_label_frame, text="partial", variable=push_save_label_var,
                                      value=2, command=push_save_label_clicked)
    partial_round_radio.pack(side="left")
    save_round_radio = tk.Radiobutton(round_label_frame, text="save", variable=push_save_label_var,
                                      value=3, command=push_save_label_clicked)
    save_round_radio.pack(side="left")

    other_state_frame = tk.Frame(window)
    other_state_frame.pack(pady=5)

    other_state_text_var = tk.StringVar()
    other_state_label = tk.Label(other_state_frame, textvariable=other_state_text_var)
    other_state_label.pack(side="left")

    details_frame = tk.Frame(window)
    details_frame.pack(pady=5)

    details_text_var = tk.StringVar()
    details_label = tk.Label(details_frame, textvariable=details_text_var, anchor="e", justify=tk.LEFT)
    details_label.pack(side="left")

    # initial value settings
    round_slider_changed(0)
    tick_slider_changed(0)

    # Start the GUI
    window.mainloop()
