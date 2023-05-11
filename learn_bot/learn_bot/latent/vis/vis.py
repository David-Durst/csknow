from learn_bot.latent.vis.draw_inference import draw_all_players
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.mining.area_cluster import *
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageDraw, ImageTk as itk

def vis(all_data_df: pd.DataFrame, pred_df: pd.DataFrame):
    make_index_column(all_data_df)
    make_index_column(pred_df)

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
    d2_img = d2_img.resize((700, 700), Image.ANTIALIAS)
    d2_img_draw = d2_img.copy()
    d2_photo_img = itk.PhotoImage(d2_img_draw)
    img_label = tk.Label(img_frame, image=d2_photo_img)
    img_label.pack(side="left")

    rounds = all_data_df.loc[:, round_id_column].unique().tolist()
    indices = []
    ticks = []
    game_ticks = []
    cur_round: int = -1
    cur_tick: int = -1
    cur_tick_index: int = -1
    selected_df: pd.DataFrame = all_data_df
    pred_selected_df: pd.DataFrame = pred_df

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
        nonlocal cur_tick, cur_tick_index, d2_img, d2_img_draw, img_label
        cur_tick_index = int(cur_tick_index_str)
        cur_index = indices[cur_tick_index]
        cur_tick = ticks[cur_tick_index]
        cur_game_tick = game_ticks[cur_tick_index]
        tick_id_text_var.set("Tick ID: " + str(cur_tick))
        tick_game_id_text_var.set("Game Tick ID: " + str(cur_game_tick))
        round_id_text_var.set(f"Round ID: {int(cur_round)}")
        d2_img_copy = d2_img.copy()
        d2_img_draw = ImageDraw.Draw(d2_img_copy)
        if len(selected_df) > 0:
            data_series = selected_df.loc[cur_index, :]
            pred_series = pred_selected_df.loc[cur_index, :]
            draw_all_players(data_series, pred_series, d2_img_draw)
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

    # state setters
    def change_round_dependent_data():
        nonlocal selected_df, pred_selected_df, cur_round, indices, ticks, game_ticks
        selected_df = all_data_df.loc[all_data_df[round_id_column] == cur_round]
        pred_selected_df = pred_df.loc[all_data_df[round_id_column] == cur_round]

        indices = selected_df.loc[:, 'index'].tolist()
        ticks = selected_df.loc[:, 'tick id'].tolist()
        game_ticks = selected_df.loc[:, 'game tick number'].tolist()
        tick_slider.configure(to=len(ticks)-1)
        tick_slider.set(0)
        tick_slider_changed(0)


    s = ttk.Style()
    s.theme_use('alt')
    s.configure('Valid.TCombobox', fieldbackground='white')
    s.configure('Invalid.TCombobox', fieldbackground='#cfcfcf')
    # creating engagement slider and label
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

    # engagegment id stepper
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

    # initial value settings
    round_slider_changed(0)
    tick_slider_changed(0)

    # Start the GUI
    window.mainloop()
