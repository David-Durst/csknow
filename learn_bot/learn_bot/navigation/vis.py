import pandas as pd
from pathlib import Path

from learn_bot.libs.temporal_column_names import TemporalIOColumnNames
from learn_bot.navigation.dataset import NavDataset
from typing import List

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

csv_outputs_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs'

non_img_df = pd.read_csv(csv_outputs_path / 'trainNav.csv')

base_vis_columns: List[str] = ["player pos", "player vis", "player vis from",
                          "distance map", "goal pos",
                          "friendly pos", "friendly vis",
                          "vis enemies", "c4 pos"]

temporal_column_names = TemporalIOColumnNames(base_vis_columns, 0, 1, 0)

nav_dataset = NavDataset(non_img_df, csv_outputs_path / 'trainNavData.tar', temporal_column_names.vis_columns)
imgs = nav_dataset.__getitem__(0)


#This creates the main window of an application
window = tk.Tk()
window.title("Nav Images")
window.geometry("800x800")
window.configure(background='grey')

# cur player's images
img = ImageTk.PhotoImage(nav_dataset.get_image_grid(0))
panel = tk.Label(window, image = img)
panel.pack(side="top")

players = list(nav_dataset.player_id.unique())

#def slider_changed(event):
#    print(slider.get())
#
#slider = tk.Scale(
#    window,
#    from_=0,
#    to=100,
#    orient='horizontal',
#    variable=current_value
#    command=slider_changed
#)

# setting combobox with color for validity
def check_combo_players(event):
    global selected_player
    value = event.widget.get()

    if value == '':
        combo_box['values'] = players
    else:
        data = []
        for item in players:
            if value.lower() in item.lower():
                data.append(item)

        combo_box['values'] = data
    if value in players:
        selected_player = value
        combo_box.configure(style='Valid.TCombobox')
    else:
        combo_box.configure(style='Invalid.TCombobox')


def update_selected_players(event):
    global selected_player
    selected_player = event.widget.get()
    combo_box.configure(style='Valid.TCombobox')


s = ttk.Style()
s.theme_use('alt')
s.configure('Valid.TCombobox', fieldbackground='white')
s.configure('Invalid.TCombobox', fieldbackground='#cfcfcf')

# creating Combobox
selected_player = players[0]
combo_box = ttk.Combobox(window, style='Valid.TCombobox')
combo_box['values'] = players
combo_box.current(0)
combo_box.bind('<KeyRelease>', check_combo_players)
combo_box.bind("<<ComboboxSelected>>", update_selected_players)
combo_box.pack()

#Start the GUI
window.mainloop()
