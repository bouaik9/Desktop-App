from tkinter import *
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from .documentation_popup import DocumentationPopup
from ..utils.input_parser import parse_adjacency_matrix, parse_lp_input

class InputScreen:
    def __init__(self, master, algorithm_name, back_callback):
        self.master = master
        self.algorithm_name = algorithm_name
        self.back_callback = back_callback
        
        self.frame = Frame(master, bg="#23272e")
        self.frame.pack(fill=BOTH, expand=True)

        self.label = Label(self.frame, text=f"Input for {algorithm_name}", bg="#23272e", fg="#f5f6fa", font=("Segoe UI", 16))
        self.label.pack(pady=10)

        self.input_area = ScrolledText(self.frame, bg="#2d313a", fg="#f5f6fa", font=("Consolas", 12), wrap=WORD)
        self.input_area.pack(pady=10, padx=10, fill=BOTH, expand=True)

        self.submit_button = Button(self.frame, text="Submit", command=self.submit, bg="#353b48", fg="#f5f6fa", font=("Segoe UI", 12))
        self.submit_button.pack(side=LEFT, padx=10, pady=10)

        self.documentation_button = Button(self.frame, text="Documentation", command=self.show_documentation, bg="#353b48", fg="#f5f6fa", font=("Segoe UI", 12))
        self.documentation_button.pack(side=LEFT, padx=10, pady=10)

        self.back_button = Button(self.frame, text="Back", command=self.back_callback, bg="#353b48", fg="#f5f6fa", font=("Segoe UI", 12))
        self.back_button.pack(side=LEFT, padx=10, pady=10)

        self.result_area = Text(self.frame, bg="#2d313a", fg="#f5f6fa", font=("Consolas", 12), wrap=WORD, height=10)
        self.result_area.pack(pady=10, padx=10, fill=BOTH, expand=True)

        self.setup_input_fields()

    def setup_input_fields(self):
        if self.algorithm_name in ["DFS", "BFS", "Prim", "Kruskal", "Graph Coloring", "Bellman", "Bellman-Ford"]:
            self.input_area.insert(END, "Enter adjacency matrix (comma-separated rows):\n")
        elif self.algorithm_name in ["Dijkstra", "Bellman-Ford"]:
            self.input_area.insert(END, "Enter adjacency matrix (comma-separated rows) and start node:\n")
        elif self.algorithm_name == "Ford-Fulkerson":
            self.input_area.insert(END, "Enter adjacency matrix (comma-separated rows), source node, and sink node:\n")
        elif self.algorithm_name in ["Revised Simplex", "Simplex", "Two-Phase"]:
            self.input_area.insert(END, "Enter constraints matrix and objective function:\n")

    def submit(self):
        input_text = self.input_area.get("1.0", END).strip()
        try:
            if self.algorithm_name in ["DFS", "BFS", "Prim", "Kruskal", "Graph Coloring", "Bellman", "Bellman-Ford"]:
                adjacency_matrix = parse_adjacency_matrix(input_text)
                result = self.run_algorithm(adjacency_matrix)
            elif self.algorithm_name in ["Dijkstra", "Bellman-Ford"]:
                adjacency_matrix, start_node = parse_adjacency_matrix(input_text, return_start=True)
                result = self.run_algorithm(adjacency_matrix, start_node)
            elif self.algorithm_name == "Ford-Fulkerson":
                adjacency_matrix, source, sink = parse_adjacency_matrix(input_text, return_source_sink=True)
                result = self.run_algorithm(adjacency_matrix, source, sink)
            elif self.algorithm_name in ["Revised Simplex", "Simplex", "Two-Phase"]:
                constraints, objective = parse_lp_input(input_text)
                result = self.run_algorithm(constraints, objective)

            self.result_area.delete("1.0", END)
            self.result_area.insert(END, result)
        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    def run_algorithm(self, *args):
        # Placeholder for actual algorithm implementations
        return "Algorithm result goes here."

    def show_documentation(self):
        DocumentationPopup(self.master, self.algorithm_name)

    def clear(self):
        self.input_area.delete("1.0", END)
        self.result_area.delete("1.0", END)