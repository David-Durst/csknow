import pandas as pd
from pathlib import Path

from learn_bot.libs.temporal_column_names import TemporalIOColumnNames
from learn_bot.navigation.dataset import NavDataset
from typing import List, Dict

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

start_time = time.perf_counter()

csv_outputs_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs'

non_img_df = pd.read_csv(csv_outputs_path / 'trainNav.csv')

base_vis_columns: List[str] = ["player pos", "player vis", "player vis from",
                          "distance map", "goal pos",
                          "friendly pos", "friendly vis",
                          "vis enemies", "c4 pos"]

temporal_column_names = TemporalIOColumnNames(base_vis_columns, 0, 1, 0)

nav_dataset = NavDataset(non_img_df, csv_outputs_path / 'trainNavData.tar', temporal_column_names.vis_columns)

end_non_img_load_time = time.perf_counter()

@dataclass(frozen=True)
class PlayerAndTick:
    player_name: str
    tick_index: int

player_tick_index_to_nav_index: Dict[PlayerAndTick, int] = {}
for nav_index, row in non_img_df.iterrows():
    player_tick_index_to_nav_index[PlayerAndTick(row['player name'], row['tick id'])] = \
        nav_index

player_dict_created_time = time.perf_counter()

#This creates the main window of an application
window = tk.Tk()
window.title("Nav Images")
window.geometry("600x650")
window.configure(background='grey')

# cur player's images
img = ImageTk.PhotoImage(nav_dataset.get_image_grid(0))
grid_img_label = tk.Label(window, image=img)
grid_img_label.pack(side="top")

first_image_creation_time = time.perf_counter()

players = list(nav_dataset.player_name.unique())
selected_player = players[0]
rounds = []
ticks = []
demo_ticks = []
cur_round: int = -1
cur_tick: int = -1
selected_df: pd.DataFrame = non_img_df

# GUI callback functions
def check_combo_players(event):
    global selected_player
    value = event.widget.get()

    if value == '':
        players_combo_box['values'] = players
    else:
        data = []
        for item in players:
            if value.lower() in item.lower():
                data.append(item)

        players_combo_box['values'] = data
    if value in players:
        selected_player = value
        players_combo_box.configure(style='Valid.TCombobox')
        change_player_dependent_data()
    else:
        players_combo_box.configure(style='Invalid.TCombobox')


def update_selected_players(event):
    global selected_player
    selected_player = event.widget.get()
    players_combo_box.configure(style='Valid.TCombobox')
    change_player_dependent_data()


def round_slider_changed(cur_round_index):
    global cur_round
    cur_round = rounds[int(cur_round_index)]
    round_id_text_var.set("Round ID: " + str(cur_round))
    change_round_dependent_data()


def tick_slider_changed(cur_tick_index):
    global cur_tick
    cur_tick = ticks[int(cur_tick_index)]
    cur_demo_tick = demo_ticks[int(cur_tick_index)]
    tick_id_text_var.set("Tick ID: " + str(cur_tick))
    tick_demo_id_text_var.set("Demo Tick ID: " + str(cur_demo_tick))
    new_img = ImageTk.PhotoImage(nav_dataset.get_image_grid(
        player_tick_index_to_nav_index[PlayerAndTick(selected_player, cur_tick)]))
    grid_img_label.configure(image=new_img)
    grid_img_label.image = new_img


def step_back_clicked():
    cur_tick_index = int(tick_slider.get())
    if cur_tick_index > 0:
        cur_tick_index -= 1
        tick_slider.set(cur_tick_index)
        tick_slider_changed(cur_tick_index)


play_active: bool = False
num_play_updates_sleeping: int = 0
def play_clicked():
    global play_active, num_play_updates_sleeping
    play_active = not play_active
    if play_active:
        play_button.configure(bg='green')
        # technically not sleeping, but need to increment by 1 so -= 1 math works out
        num_play_updates_sleeping += 1
        play_update()
    else:
        play_button.configure(bg=orig_player_button_color)


def play_update():
    global num_play_updates_sleeping
    num_play_updates_sleeping -= 1
    if play_active and num_play_updates_sleeping == 0:
        step_forward_clicked()
        num_play_updates_sleeping += 1
        play_button.after(250, play_update)


def step_forward_clicked():
    cur_tick_index = int(tick_slider.get())
    if cur_tick_index < len(ticks) - 1:
        cur_tick_index += 1
        tick_slider.set(cur_tick_index)
        tick_slider_changed(cur_tick_index)


# state setters
def change_player_dependent_data():
    global selected_df, rounds
    selected_df = non_img_df.loc[non_img_df['player name'] == selected_player]
    rounds = list(selected_df.loc[:, 'round id'].unique())
    round_slider.configure(to=len(rounds)-1)
    round_slider.set(0)
    round_slider_changed(0)


def change_round_dependent_data():
    global selected_df, cur_round, ticks, demo_ticks
    selected_df = non_img_df.loc[(non_img_df['player name'] == selected_player) &
                                 (non_img_df['round id'] == cur_round)]
    ticks = list(selected_df.loc[:, 'tick id'])
    demo_ticks = list(selected_df.loc[:, 'demo tick id'])
    tick_slider.configure(to=len(ticks)-1)
    tick_slider.set(0)
    tick_slider_changed(0)


s = ttk.Style()
s.theme_use('alt')
s.configure('Valid.TCombobox', fieldbackground='white')
s.configure('Invalid.TCombobox', fieldbackground='#cfcfcf')

# creating player combobox
players_combo_box = ttk.Combobox(window, style='Valid.TCombobox')
players_combo_box['values'] = players
players_combo_box.current(0)
players_combo_box.bind('<KeyRelease>', check_combo_players)
players_combo_box.bind("<<ComboboxSelected>>", update_selected_players)
players_combo_box.pack(pady=5)

# creating round slider and label
round_frame = tk.Frame(window)
round_frame.pack(pady=5)

round_id_text_var = tk.StringVar()
round_id_label = tk.Label(round_frame, textvariable=round_id_text_var)
round_id_label.pack(side="left")
round_slider = tk.Scale(
    round_frame,
    from_=0,
    to=100,
    orient='horizontal',
    showvalue=0,
    length=400,
    command=round_slider_changed
)
round_slider.pack(side="left")

# creating tick slider and label
tick_id_frame = tk.Frame(window)
tick_id_frame.pack(pady=5)

tick_id_text_var = tk.StringVar()
tick_id_label = tk.Label(tick_id_frame, textvariable=tick_id_text_var)
tick_id_label.pack(side="left")

tick_demo_id_text_var = tk.StringVar()
tick_demo_id_label = tk.Label(tick_id_frame, textvariable=tick_demo_id_text_var)
tick_demo_id_label.pack(side="left")

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
tick_step_frame = tk.Frame(window)
tick_step_frame.pack(pady=5)
back_step_button = tk.Button(tick_step_frame, text="⏪", command=step_back_clicked)
back_step_button.pack(side="left")
play_button = tk.Button(tick_step_frame, text="⏯", command=play_clicked)
orig_player_button_color = play_button.cget("background")
play_button.pack(side="left")
forward_step_button = tk.Button(tick_step_frame, text="⏩", command=step_forward_clicked)
forward_step_button.pack(side="left")

# initial value settings
change_player_dependent_data()
round_slider_changed(0)
tick_slider_changed(0)

ready_time = time.perf_counter()

print(f"non img load time {end_non_img_load_time - start_time: 0.4f}")
print(f"player dict creation time {player_dict_created_time - end_non_img_load_time: 0.4f}")
print(f"first image creation time {first_image_creation_time - player_dict_created_time: 0.4f}")
print(f"ready time {ready_time - first_image_creation_time: 0.4f}")
print(f"total time {ready_time - start_time: 0.4f}")

# Start the GUI
window.mainloop()
