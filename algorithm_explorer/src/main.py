# This file is the entry point of the application. It initializes the Tkinter main window, sets up the main screen for algorithm selection, and manages navigation between screens.

import tkinter as tk
from ui.main_screen import MainScreen
from ui.theme import apply_theme

class AlgorithmExplorerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Explorer")
        self.root.geometry("800x600")
        apply_theme(self.root)

        self.main_screen = MainScreen(self.root, self.switch_to_input_screen)
        self.main_screen.pack(fill=tk.BOTH, expand=True)

    def switch_to_input_screen(self, algorithm):
        self.main_screen.pack_forget()
        from ui.input_screen import InputScreen
        self.input_screen = InputScreen(self.root, algorithm, self.switch_to_main_screen)
        self.input_screen.pack(fill=tk.BOTH, expand=True)

    def switch_to_main_screen(self):
        self.input_screen.pack_forget()
        self.main_screen.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmExplorerApp(root)
    root.mainloop()