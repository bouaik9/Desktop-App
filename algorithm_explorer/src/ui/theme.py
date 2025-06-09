# This file defines the dark theme for the application, including color constants and font settings to ensure a consistent look and feel across the UI.

import tkinter as tk
from tkinter import ttk

BACKGROUND_COLOR = "#23272e"
FOREGROUND_COLOR = "#f5f6fa"
ACCENT_COLOR = "#4f8cff"
BUTTON_BACKGROUND_COLOR = "#353b48"
ENTRY_BACKGROUND_COLOR = "#2d313a"
FONT_NORMAL = "Segoe UI"
FONT_INPUT = "Consolas"

def apply_theme(root):
    root.configure(bg=BACKGROUND_COLOR)


    style = ttk.Style()

    style.configure("TLabel", background=BACKGROUND_COLOR, foreground=FOREGROUND_COLOR, font=FONT_NORMAL)
    style.configure("TButton", background=BUTTON_BACKGROUND_COLOR, foreground=FOREGROUND_COLOR, font=FONT_NORMAL, borderwidth=0, highlightthickness=0)
    style.map("TButton",
              background=[("active", ACCENT_COLOR)],
              foreground=[("active", FOREGROUND_COLOR)])

    style.configure("TEntry",
                    fieldbackground=ENTRY_BACKGROUND_COLOR,
                    foreground=FOREGROUND_COLOR,
                    font=FONT_INPUT)