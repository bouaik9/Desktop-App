from tkinter import Frame, Label, Listbox, Button, Scrollbar, Toplevel, messagebox

class MainScreen(Frame):
    def __init__(self, master, algorithm_selection_callback):
        super().__init__(master)
        self.master = master
        self.algorithm_selection_callback = algorithm_selection_callback
        self.init_ui()

    def init_ui(self):
        self.configure(bg="#23272e")

        title = Label(self, text="Select an Algorithm", bg="#23272e", fg="#f5f6fa", font=("Segoe UI", 16))
        title.pack(pady=10)

        self.algorithm_listbox = Listbox(self, bg="#2d313a", fg="#f5f6fa", font=("Consolas", 12), selectbackground="#4f8cff")
        self.algorithm_listbox.pack(pady=10, padx=10, fill='both', expand=True)

        algorithms = [
            "LP: Revised Simplex Method",
            "Depth-First Search (DFS)",
            "Bellman’s Algorithm",
            "Prim’s Algorithm",
            "LP: Simplex Algorithm",
            "Breadth-First Search (BFS)",
            "Dijkstra’s Algorithm",
            "Kruskal’s Algorithm",
            "LP: Two-Phase Method",
            "Graph Coloring",
            "Bellman-Ford Algorithm",
            "Ford-Fulkerson Algorithm"
        ]

        for algorithm in algorithms:
            self.algorithm_listbox.insert('end', algorithm)

        scrollbar = Scrollbar(self.algorithm_listbox)
        scrollbar.pack(side='right', fill='y')
        self.algorithm_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.algorithm_listbox.yview)

        select_button = Button(self, text="Select", command=self.on_select, bg="#4f8cff", fg="#f5f6fa", font=("Segoe UI", 12))
        select_button.pack(pady=10)

        self.pack(fill='both', expand=True)

    def on_select(self):
        selected_index = self.algorithm_listbox.curselection()
        if selected_index:
            selected_algorithm = self.algorithm_listbox.get(selected_index)
            self.algorithm_selection_callback(selected_algorithm)
        else:
            messagebox.showwarning("Selection Error", "Please select an algorithm.")