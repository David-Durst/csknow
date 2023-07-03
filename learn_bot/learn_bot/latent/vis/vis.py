from typing import Set, Callable

from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.train import good_retake_rounds_path
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.mining.area_cluster import *
from learn_bot.latent.vis.draw_inference import draw_all_players, minimapWidth, minimapHeight
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageDraw, ImageTk as itk


def get_rounds_for_cur_hdf5(loaded_model: LoadedModel) -> List[int]:
    return loaded_model.cur_loaded_df.loc[:, round_id_column].unique().tolist()


def index_cur_hdf5(loaded_model: LoadedModel):
    make_index_column(loaded_model.cur_loaded_df)
    make_index_column(loaded_model.cur_inference_df)


def vis(loaded_model: LoadedModel, inference_fn: Callable[[LoadedModel], None]):
    inference_fn(loaded_model)
    index_cur_hdf5(loaded_model)

    #This creates the main window of an application
    window = tk.Tk()
    window.title("Delta Position Model")
    window.resizable(width=True, height=True)
    window.configure(background='grey')

    your_font = font.nametofont("TkDefaultFont")  # Get default font value into Font object
    your_font.actual()

    img_frame = tk.Frame(window)
    img_frame.pack(pady=5)
    d2_img = Image.open(d2_radar_path)
    d2_img = d2_img.resize((minimapWidth, minimapHeight), Image.ANTIALIAS)
    d2_img_draw = d2_img.copy()
    d2_photo_img = itk.PhotoImage(d2_img_draw)
    img_label = tk.Label(img_frame, image=d2_photo_img)
    img_label.pack(side="left")

    rounds = get_rounds_for_cur_hdf5(loaded_model)
    indices = []
    ticks = []
    game_ticks = []
    cur_round: int = -1
    cur_tick: int = -1
    cur_tick_index: int = -1
    selected_df: pd.DataFrame = loaded_model.cur_loaded_df
    pred_selected_df: pd.DataFrame = loaded_model.cur_inference_df
    draw_max: bool = True
    good_retake_rounds: Set[int] = set()

    def hdf5_id_update():
        nonlocal rounds, cur_round
        loaded_model.cur_hdf5_index = int(new_hdf5_id_entry.get())
        loaded_model.load_cur_hdf5_as_pd()
        inference_fn(loaded_model)
        index_cur_hdf5(loaded_model)
        rounds = get_rounds_for_cur_hdf5(loaded_model)
        cur_round = rounds[0]
        round_slider.set(0)
        round_slider_changed(0)

    def round_slider_changed(cur_round_index):
        nonlocal cur_round
        cur_round = rounds[int(cur_round_index)]
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
        nonlocal cur_tick, cur_tick_index, d2_img, d2_img_draw, img_label, draw_max
        cur_tick_index = int(cur_tick_index_str)
        cur_index = indices[cur_tick_index]
        cur_tick = ticks[cur_tick_index]
        cur_game_tick = game_ticks[cur_tick_index]
        hdf5_id_text_var.set(f"Cur HDF5 Id: {loaded_model.cur_hdf5_index}, ")
        tick_id_text_var.set("Tick ID: " + str(cur_tick))
        tick_game_id_text_var.set("Game Tick ID: " + str(cur_game_tick))
        round_id_text_var.set(f"Round ID: {int(cur_round)}, Round Number: {selected_df.loc[cur_index, 'round number']}")
        d2_img_copy = d2_img.copy().convert("RGBA")
        d2_overlay_im = Image.new("RGBA", d2_img_copy.size, (255, 255, 255, 0))
        d2_img_draw = ImageDraw.Draw(d2_overlay_im)
        if len(selected_df) > 0:
            data_series = selected_df.loc[cur_index, :]
            pred_series = pred_selected_df.loc[cur_index, :]
            other_state_text_var.set(f"Planted A {data_series[c4_plant_a_col]}, "
                                     f"Planted B {data_series[c4_plant_b_col]}, "
                                     f"Not Planted {data_series[c4_not_planted_col]}, "
                                     f"C4 Pos ({data_series[c4_pos_cols[0]]}, {data_series[c4_pos_cols[1]]}, {data_series[c4_pos_cols[2]]})")
            players_to_draw_str = player_distributions_var.get()
            if players_to_draw_str == "*":
                players_to_draw = list(range(0, len(specific_player_place_area_columns)))
            else:
                players_to_draw = [int(p) for p in players_to_draw_str.split(",")]
            players_str = draw_all_players(data_series, pred_series, d2_img_draw, draw_max, players_to_draw)
            details_text_var.set(players_str)
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
            play_button.after(250, play_update)


    def step_forward_clicked():
        nonlocal cur_tick_index
        if cur_tick_index < len(ticks) - 1:
            cur_tick_index += 1
            tick_slider.set(cur_tick_index)
            tick_slider_changed(cur_tick_index)


    def toggle_distribution_clicked():
        nonlocal draw_max
        draw_max = not draw_max
        tick_slider_changed(cur_tick_index)

    def good_retake_clicked():
        if good_retake_var.get():
            good_retake_rounds.add(cur_round)
        else:
            good_retake_rounds.remove(cur_round)
        with open(good_retake_rounds_path, "w") as f:
            f.write(str(good_retake_rounds) + "\n")

    # state setters
    def change_round_dependent_data():
        nonlocal selected_df, pred_selected_df, cur_round, indices, ticks, game_ticks
        selected_df = loaded_model.cur_loaded_df.loc[loaded_model.cur_loaded_df[round_id_column] == cur_round]
        pred_selected_df = loaded_model.cur_inference_df.loc[loaded_model.cur_loaded_df[round_id_column] == cur_round]

        indices = selected_df.loc[:, 'index'].tolist()
        ticks = selected_df.loc[:, 'tick id'].tolist()
        game_ticks = selected_df.loc[:, 'game tick number'].tolist()
        tick_slider.configure(to=len(ticks)-1)
        tick_slider.set(0)
        tick_slider_changed(0)
        good_retake_var.set(cur_round in good_retake_rounds)


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

    good_retake_var = tk.BooleanVar(value=False)
    good_retake_checkbutton = tk.Checkbutton(distribution_control_frame, text='Good Retake', variable=good_retake_var,
                                             onvalue=True, offvalue=False, command=good_retake_clicked)
    good_retake_checkbutton.pack(side="left")

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
