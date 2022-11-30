import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from typing import Optional, Tuple


class ChildWindow:
    window: Optional[tk.Toplevel]
    canvas: Optional[FigureCanvasTkAgg]
    figure: Optional[Figure]

    def __init__(self):
        self.window = None
        self.canvas = None
        self.figure = None

    def is_valid(self):
        return self.window is not None

    def initialize(self, parent_window: tk.Tk, figsize: Tuple[float, float], dpi: int = 100) -> bool:
        if not self.is_valid():
            self.figure = Figure(figsize=figsize, dpi=dpi)

            self.window = tk.Toplevel(parent_window)
            self.window.protocol('WM_DELETE_WINDOW', self.remove_window)

            self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.toolbar = NavigationToolbar2Tk(self.canvas, self.window, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            return True
        else:
            return False

    def remove_window(self):
        self.window.destroy()
        self.window = None
