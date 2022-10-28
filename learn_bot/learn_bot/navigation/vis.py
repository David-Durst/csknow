import pandas as pd
from pathlib import Path

from learn_bot.libs.temporal_column_names import TemporalIOColumnNames
from learn_bot.navigation.dataset import NavDataset
from typing import List

import tkinter as tk
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
window.title("Test Image")
window.geometry("800x800")
window.configure(background='grey')

img = ImageTk.PhotoImage(nav_dataset.get_image_grid(0))
panel = tk.Label(window, image = img)
panel.pack(side="bottom", fill="both", expand="yes")

#Start the GUI
window.mainloop()

x = 1
