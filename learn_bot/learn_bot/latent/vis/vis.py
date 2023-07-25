from typing import Set, Callable

from learn_bot.latent.analyze.process_trajectory_comparison import set_pd_print_options
from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.train import default_selected_retake_rounds_path
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
    print(f"num rounds {len(rounds)}")
    indices = []
    ticks = []
    game_ticks = []
    cur_round: int = -1
    cur_tick: int = -1
    cur_tick_index: int = -1
    selected_df: pd.DataFrame = loaded_model.cur_loaded_df
    pred_selected_df: pd.DataFrame = loaded_model.cur_inference_df
    id_df: pd.DataFrame = loaded_model.get_cur_id_df()
    draw_pred: bool = True
    draw_max: bool = True
    selected_retake_rounds: Set[int] = set()

    set_pd_print_options()

    def hdf5_id_update():
        nonlocal rounds, cur_round
        loaded_model.cur_hdf5_index = int(new_hdf5_id_entry.get())
        loaded_model.load_cur_hdf5_as_pd()
        inference_fn(loaded_model)
        index_cur_hdf5(loaded_model)
        rounds = get_rounds_for_cur_hdf5(loaded_model)
        cur_round = rounds[0]
        print(f"num rounds {len(rounds)}")
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
        if selected_df.empty:
            print("why empty selected_df?")
            return
        cur_tick_index = int(cur_tick_index_str)
        cur_index = indices[cur_tick_index]
        cur_tick = ticks[cur_tick_index]
        cur_game_tick = game_ticks[cur_tick_index]

        hdf5_id_text_var.set(f"Predicted Cur HDF5 Id: {loaded_model.get_cur_hdf5_filename()} - {loaded_model.cur_hdf5_index} / "
                             f"{len(loaded_model.dataset.data_hdf5s) - 1}, ")
        tick_id_text_var.set("Tick ID: " + str(cur_tick))
        data_series = selected_df.loc[cur_index, :]
        pred_series = pred_selected_df.loc[cur_index, :]
        tick_game_id_text_var.set(f"Game Tick ID: {cur_game_tick}")
        extra_round_data_str = ""
        if get_similarity_column(0) in id_df.columns:
            extra_round_data_str = f"similarity 0: {id_df.loc[cur_index, get_similarity_column(0)]}, similarity 1: {id_df.loc[cur_index, get_similarity_column(1)]}"
        round_id_text_var.set(f"Round ID: {int(cur_round)}, Round Number: {selected_df.loc[cur_index, 'round number']}, {extra_round_data_str}")
        other_state_text_var.set(f"Planted A {data_series[c4_plant_a_col]}, "
                                 f"Planted B {data_series[c4_plant_b_col]}, "
                                 f"Not Planted {data_series[c4_not_planted_col]}, "
                                 f"C4 Pos ({data_series[c4_pos_cols[0]]:.2f}, {data_series[c4_pos_cols[1]]:.2f}, {data_series[c4_pos_cols[2]]:.2f}),"
                                 f"C4 Time Left Percent {data_series[c4_time_left_percent[0]]:.2f}")

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
                player_to_color[i] = (4, 190, 196)
                player_to_color[i+max_enemies] = (187, 142, 52)
        players_str = draw_all_players(data_series, pred_series, d2_img_draw, draw_max, players_to_draw,
                                       player_to_color=player_to_color, draw_only_pos=not draw_pred)

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

    def pred_toggle_clicked():
        nonlocal draw_pred
        draw_pred = not draw_pred
        tick_slider_changed(cur_tick_index)

    def print_row_clicked():
        cur_index = indices[cur_tick_index]
        non_delta_pos_cols = []
        for col in selected_df.columns:
            if "delta pos" not in col:
                non_delta_pos_cols.append(col)
        data_series = selected_df.loc[cur_index, non_delta_pos_cols]
        print(data_series)


    def toggle_distribution_clicked():
        nonlocal draw_max
        draw_max = not draw_max
        tick_slider_changed(cur_tick_index)

    def select_all_retake_rounds():
        nonlocal selected_retake_rounds
        selected_retake_rounds = set(rounds)
        selected_retake_var.set(True)
        with open(selected_retake_rounds_path_var.get(), "w") as f:
            f.write(str(selected_retake_rounds) + "\n")

    def load_selected_retake_rounds():
        nonlocal selected_retake_rounds
        with open(selected_retake_rounds_path_var.get(), "r") as f:
            selected_retake_rounds_str = f.readline().strip()
            selected_retake_rounds = eval(selected_retake_rounds_str)
            selected_retake_var.set(cur_round in selected_retake_rounds)

    def selected_retake_clicked():
        if selected_retake_var.get():
            selected_retake_rounds.add(cur_round)
        else:
            selected_retake_rounds.remove(cur_round)
        with open(selected_retake_rounds_path_var.get(), "w") as f:
            f.write(str(selected_retake_rounds) + "\n")

    # state setters
    def change_round_dependent_data():
        nonlocal selected_df, id_df, pred_selected_df, cur_round, indices, ticks, game_ticks
        selected_df = loaded_model.cur_loaded_df.loc[loaded_model.cur_loaded_df[round_id_column] == cur_round]
        id_df = loaded_model.get_cur_id_df()
        pred_selected_df = loaded_model.cur_inference_df.loc[loaded_model.cur_loaded_df[round_id_column] == cur_round]

        indices = selected_df.loc[:, 'index'].tolist()
        ticks = selected_df.loc[:, 'tick id'].tolist()
        game_ticks = selected_df.loc[:, 'game tick number'].tolist()
        tick_slider.configure(to=len(ticks)-1)
        tick_slider.set(0)
        tick_slider_changed(0)
        selected_retake_var.set(cur_round in selected_retake_rounds)


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

    round_label_frame = tk.Frame(window)
    round_label_frame.pack(pady=5)
    selected_retake_rounds_path_var = tk.StringVar(value=default_selected_retake_rounds_path)
    selected_retake_rounds_entry = tk.Entry(round_label_frame, width=50, textvariable=selected_retake_rounds_path_var)
    selected_retake_rounds_entry.pack(side="left")
    select_all_retake_rounds_button = tk.Button(round_label_frame, text="select all retake rounds", command=select_all_retake_rounds)
    select_all_retake_rounds_button.pack(side="left")
    load_selected_retake_rounds_button = tk.Button(round_label_frame, text="load selected retake rounds", command=load_selected_retake_rounds)
    load_selected_retake_rounds_button.pack(side="left")
    selected_retake_var = tk.BooleanVar(value=False)
    selected_retake_checkbutton = tk.Checkbutton(round_label_frame, text='Selected Retake', variable=selected_retake_var,
                                             onvalue=True, offvalue=False, command=selected_retake_clicked)
    selected_retake_checkbutton.pack(side="left")

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
