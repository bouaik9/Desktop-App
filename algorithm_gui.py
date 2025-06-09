import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
from collections import deque
import numpy as np
from utilities.graph_coloring_draw import graph_coloring_draw
from utilities.graph_from_matrix import draw_graph_from_matrix
from Algorithms import *

import threading
import webbrowser

# Algorithm metadata
ALGORITHMS = {
    "Programmation Linéaire": {
        "LP: Méthode du Simplex Révisée": {
            "description": "An efficient version of the simplex method for solving linear programming problems.",
            "input_type": "matrix"
        },
        "LP: Algorithme de Simplex": {
            "description": "A popular algorithm for linear programming problems.",
            "input_type": "matrix"
        },
        "LP: Méthode en Deux Phases": {
            "description": "Solves linear programming problems with artificial variables.",
            "input_type": "matrix"
        }
    },
    "Théorie des Graphes": {
        "Depth-First Search (DFS)": {
            "description": "A graph traversal algorithm that explores as far as possible along each branch before backtracking.",
            "input_type": "adjacency"
        },
        "Algorithme de Bellman": {
            "description": "Finds shortest paths from a single source vertex to all other vertices in a weighted digraph.",
            "input_type": "adjacency"
        },
        "Algorithme de Prim": {
            "description": "Finds a minimum spanning tree for a weighted undirected graph.",
            "input_type": "adjacency"
        },
        "Breadth-First Search (BFS)": {
            "description": "A graph traversal algorithm that explores all neighbors at the present depth before moving on.",
            "input_type": "adjacency"
        },
        "Algorithme de Dijkstra": {
            "description": "Finds the shortest path between nodes in a graph.",
            "input_type": "adjacency"
        },
        "Algorithme de Kruskal": {
            "description": "Finds a minimum spanning tree for a connected weighted graph.",
            "input_type": "adjacency"
        },
        "Coloration de Graphes": {
            "description": "Assigns colors to graph vertices so that no two adjacent vertices share the same color.",
            "input_type": "adjacency"
        },
        "Algorithme de Bellman-Ford": {
            "description": "Computes shortest paths from a single source vertex to all other vertices in a weighted digraph.",
            "input_type": "adjacency"
        },
        "Algorithme de Ford-Fulkerson": {
            "description": "Computes the maximum flow in a flow network.",
            "input_type": "adjacency"
        }
    }
}

# Color scheme (dark theme)
BG_COLOR = "#23272e"
FG_COLOR = "#f5f6fa"
ACCENT_COLOR = "#4f8cff"
BUTTON_BG = "#353b48"
BUTTON_FG = "#f5f6fa"
ENTRY_BG = "#2d313a"
ENTRY_FG = "#f5f6fa"

# --- Algorithm Implementations ---

def parse_matrix_input(text):
    """
    Parse matrix input: each line is a row, values separated by spaces or commas.
    Returns a list of lists of floats.
    """
    matrix = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        row = [float(x) for x in line.replace(',', ' ').split()]
        matrix.append(row)
    return matrix

def parse_adjacency_input(text):
    """
    Parse adjacency matrix input: each line is a row, values separated by spaces or commas.
    Returns a list of lists of floats.
    """
    return parse_matrix_input(text)

def parse_graph_edges(text):
    """
    Parse edge list: each line is "u v w" (from, to, weight)
    Returns a list of (u, v, w)
    """
    edges = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) == 2:
            u, v = map(int, parts)
            w = 1
        else:
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
        edges.append((u, v, w))
    return edges




def bellmans_algorithm(adj, start=0):
    # Alias for Bellman-Ford
    return bellman_ford(adj, start)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Algorithm Explorer")
        self.geometry("800x800")
        self.configure(bg=BG_COLOR)
        self.resizable(False, False)
        self.selected_domain = None
        self.selected_algorithm = None

        self.container = tk.Frame(self, bg=BG_COLOR)
        self.container.pack(fill="both", expand=True)

        self.show_domain_selection_view()

    def show_domain_selection_view(self):
        for widget in self.container.winfo_children():
            widget.destroy()
        DomainSelectionView(self.container, self).pack(fill="both", expand=True)

    def show_algorithm_selection_view(self, domain):
        self.selected_domain = domain
        for widget in self.container.winfo_children():
            widget.destroy()
        AlgorithmSelectionView(self.container, self, domain).pack(fill="both", expand=True)

    def show_input_view(self, algorithm_name):
        self.selected_algorithm = algorithm_name
        for widget in self.container.winfo_children():
            widget.destroy()
        AlgorithmInputView(self.container, self, algorithm_name).pack(fill="both", expand=True)

class DomainSelectionView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_COLOR)
        self.controller = controller

        title_font = font.Font(family="Segoe UI", size=22, weight="bold")
        label = tk.Label(self, font=title_font,
                         bg=BG_COLOR, fg=ACCENT_COLOR, pady=20)
        label.pack()

        button_frame = tk.Frame(self, bg=BG_COLOR)
        button_frame.pack(pady=180)

        btn_font = font.Font(family="Segoe UI", size=12)
        for domain in ALGORITHMS:
            btn = tk.Button(
                button_frame,
                text=domain,
                font=btn_font,
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=ACCENT_COLOR,
                activeforeground=FG_COLOR,
                relief="flat",
                padx=12, pady=8,
                anchor="w",
                width=32,
                command=lambda d=domain: controller.show_algorithm_selection_view(d)
            )
            btn.pack(fill="x", pady=4, padx=20)

class AlgorithmSelectionView(tk.Frame):
    def __init__(self, parent, controller, domain):
        super().__init__(parent, bg=BG_COLOR)
        self.controller = controller
        self.domain = domain

        title_font = font.Font(family="Segoe UI", size=18, weight="bold")
        label = tk.Label(self, text=domain, font=title_font,
                         bg=BG_COLOR, fg=ACCENT_COLOR, pady=16)
        label.pack()

        button_frame = tk.Frame(self, bg=BG_COLOR)
        button_frame.pack(pady=10)

        btn_font = font.Font(family="Segoe UI", size=12)
        for algo in ALGORITHMS[domain]:
            btn = tk.Button(
                button_frame,
                text=algo,
                font=btn_font,
                bg=BUTTON_BG,
                fg=BUTTON_FG,
                activebackground=ACCENT_COLOR,
                activeforeground=FG_COLOR,
                relief="flat",
                padx=12, pady=8,
                anchor="w",
                width=32,
                command=lambda a=algo: controller.show_input_view(a)
            )
            btn.pack(fill="x", pady=4, padx=20)

        back_btn = tk.Button(
            self, text="précédent",
            font=("Segoe UI", 11),
            bg=BUTTON_BG, fg=BUTTON_FG,
            activebackground=ACCENT_COLOR, activeforeground=FG_COLOR,
            relief="flat", padx=10, pady=6,
            command=controller.show_domain_selection_view
        )
        back_btn.pack(pady=12)

class AlgorithmInputView(tk.Frame):
    def __init__(self, parent, controller, algorithm_name):
        super().__init__(parent, bg=BG_COLOR)
        self.controller = controller
        self.algorithm_name = algorithm_name
        self.domain = None

        # Find the domain the algorithm belongs to
        for domain, algos in ALGORITHMS.items():
            if algorithm_name in algos:
                self.domain = domain
                break

        title_font = font.Font(family="Segoe UI", size=18, weight="bold")
        label = tk.Label(self, text=algorithm_name, font=title_font,
                         bg=BG_COLOR, fg=ACCENT_COLOR, pady=16)
        label.pack()

        # Input area
        input_label = tk.Label(self, text="Saisissez les données d'entrée",
                               font=("Segoe UI", 12), bg=BG_COLOR, fg=FG_COLOR)
        input_label.pack(anchor="w", padx=32)

        self.extra_inputs = {}

        # Adjacency matrix input
        self.input_text = tk.Text(self, height=8, width=48, font=("Consolas", 12),
                                  bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, bd=1, relief="solid")
        self.input_text.pack(padx=32, pady=8)

        # Additional fields for certain algorithms
        algo = self.algorithm_name
        if algo in ["Algorithme de Dijkstra", "Algorithme de Bellman-Ford", "Algorithme de Bellman"]:
            # Start node input
            start_frame = tk.Frame(self, bg=BG_COLOR)
            start_frame.pack(anchor="w", padx=32, pady=(0, 8))
            tk.Label(start_frame, text="Noeud de départ:", font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
            self.extra_inputs['start'] = tk.Entry(start_frame, width=5, font=("Consolas", 12), bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
            self.extra_inputs['start'].pack(side="left", padx=(8, 0))
            self.extra_inputs['start'].insert(0, "0")
        elif algo == "Algorithme de Ford-Fulkerson":
            # Source and sink input
            ss_frame = tk.Frame(self, bg=BG_COLOR)
            ss_frame.pack(anchor="w", padx=32, pady=(0, 8))
            tk.Label(ss_frame, text="Source:", font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
            self.extra_inputs['source'] = tk.Entry(ss_frame, width=5, font=("Consolas", 12), bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
            self.extra_inputs['source'].pack(side="left", padx=(8, 16))
            self.extra_inputs['source'].insert(0, "0")
            tk.Label(ss_frame, text="Sink:", font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
            self.extra_inputs['sink'] = tk.Entry(ss_frame, width=5, font=("Consolas", 12), bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
            self.extra_inputs['sink'].pack(side="left", padx=(8, 0))
            self.extra_inputs['sink'].insert(0, "1")
        elif algo in ["LP: Algorithme de Simplex", "LP: Méthode du Simplex Révisée", "LP: Méthode en Deux Phases"]:
            # Show a hint for matrix input
            hint = (
                "Format d'entrée de la matrice :\n"
                "Chaque ligne = contrainte ou objectif, valeurs séparées par un espace ou une virgule.\n"
                "Dernière ligne = coefficients de l'objectif.\n"
                "Dernière colonne = membre de droite (b).\n"
                "Exemple :\n"
                "1 1 1 40\n"
                "2 1 0 60\n"
                "0 1 1 50\n"
                "3 2 1 0"
            )
            hint_label = tk.Label(self, text=hint, font=("Consolas", 9), bg=BG_COLOR, fg="#aaa", justify="left")
            hint_label.pack(anchor="w", padx=32, pady=(0, 8))

        # Button row
        btn_frame = tk.Frame(self, bg=BG_COLOR)
        btn_frame.pack(pady=12)

        doc_btn = tk.Button(
            btn_frame, text="Documentation",
            font=("Segoe UI", 11),
            bg=BUTTON_BG, fg=BUTTON_FG,
            activebackground=ACCENT_COLOR, activeforeground=FG_COLOR,
            relief="flat", padx=10, pady=6,
            command=self.show_documentation
        )
        doc_btn.grid(row=0, column=0, padx=8)

        submit_btn = tk.Button(
            btn_frame, text="Submit",
            font=("Segoe UI", 11, "bold"),
            bg=ACCENT_COLOR, fg=FG_COLOR,
            activebackground=BUTTON_BG, activeforeground=FG_COLOR,
            relief="flat", padx=16, pady=6,
            command=self.process_input
        )
        submit_btn.grid(row=0, column=1, padx=8)

        back_btn = tk.Button(
            btn_frame, text="précédent",
            font=("Segoe UI", 11),
            bg=BUTTON_BG, fg=BUTTON_FG,
            activebackground=ACCENT_COLOR, activeforeground=FG_COLOR,
            relief="flat", padx=10, pady=6,
            command=lambda: controller.show_algorithm_selection_view(self.domain)
        )
        back_btn.grid(row=0, column=2, padx=8)

        # Result label
        self.result_label = tk.Label(self, text="", font=("Segoe UI", 12, "italic"),
                                    bg=BG_COLOR, fg=ACCENT_COLOR, wraplength=400, pady=10)
        self.result_label.pack()

    def show_documentation(self):
        # Mapping of algorithms to documentation URLs
        doc_links = {
            "LP: Méthode du Simplex Révisée": "https://pageperso.lis-lab.fr/christophe.gonzales/teaching/optimisation/cours/cours04.pdf",
            "LP: Algorithme de Simplex": "https://youtu.be/i8vnEZi3e4A?si=tCHFjZVCbnGGtb1l",
            "LP: Méthode en Deux Phases": "https://youtu.be/_wnqe5_CLU0?si=Cq96OVW1Sokjtvu2",
            "Depth-First Search (DFS)": "https://youtu.be/pcKY4hjDrxk?si=x26RxQWB-f3Ly_qJ",
            "Algorithme de Bellman": "https://youtu.be/obWXjtg0L64?si=6jN-p6V-iXna_-lC",
            "Algorithme de Prim": "https://youtu.be/4ZlRH0eK-qQ?si=Cm0zVd_c0CYYMNAZ",
            "Breadth-First Search (BFS)": "https://youtu.be/pcKY4hjDrxk?si=RXoqBUh0_CzCgTVC",
            "Algorithme de Dijkstra": "https://fr.wikipedia.org/wiki/Algorithme_de_Dijkstra",
            "Algorithme de Kruskal": "https://youtu.be/4ZlRH0eK-qQ?si=Cm0zVd_c0CYYMNAZ",
            "Coloration de Graphes": "https://youtu.be/3VeQhNF5-rE?si=BVDFIBQa531FUXrp",
            "Algorithme de Bellman-Ford": "https://youtu.be/obWXjtg0L64?si=6jN-p6V-iXna_-lC",
            "Algorithme de Ford-Fulkerson": "https://youtu.be/Tl90tNtKvxs?si=8ELb0K0exRQ1KNfZ",
        }
        url = doc_links.get(self.algorithm_name)
        if url:
            webbrowser.open(url)
        else:
            messagebox.showinfo("Documentation", "Aucune documentation en ligne disponible pour cet algorithme.")

    def process_input(self):
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("Input Required", "Please enter input data for the algorithm.")
            return

        # Find the domain the algorithm belongs to
        for domain, algos in ALGORITHMS.items():
            if self.algorithm_name in algos:
                input_type = ALGORITHMS[domain][self.algorithm_name]["input_type"]
                break

        try:
            if input_type == "matrix":
                matrix = parse_matrix_input(user_input)
                if self.algorithm_name == "LP: Algorithme de Simplex":
                    result = simplex_algorithm(matrix)
                elif self.algorithm_name == "LP: Méthode du Simplex Révisée":
                    result = revised_simplex_method(matrix)
                elif self.algorithm_name == "LP: Méthode en Deux Phases":
                    result = two_phase_method(matrix)
                else:
                    result = "Unknown matrix algorithm."
            elif input_type == "adjacency":
                adj = parse_adjacency_input(user_input)
                algo = self.algorithm_name
                if algo == "Depth-First Search (DFS)":
                    result = dfs(adj)
                elif algo == "Breadth-First Search (BFS)":
                    result = bfs(adj)
                elif algo == "Algorithme de Dijkstra":
                    start = int(self.extra_inputs['start'].get()) if 'start' in self.extra_inputs else 0
                    result = dijkstra(adj, start)
                elif algo == "Algorithme de Bellman-Ford":
                    start = int(self.extra_inputs['start'].get()) if 'start' in self.extra_inputs else 0
                    result = bellman_ford(adj, start)
                elif algo == "Algorithme de Bellman":
                    start = int(self.extra_inputs['start'].get()) if 'start' in self.extra_inputs else 0
                    result = bellmans_algorithm(adj, start)
                elif algo == "Algorithme de Prim":
                    result = prim(adj)
                elif algo == "Algorithme de Kruskal":
                    result = kruskal(adj)
                elif algo == "Coloration de Graphes":
                    result = graph_coloring(adj)
                elif algo == "Algorithme de Ford-Fulkerson":
                    source = int(self.extra_inputs['source'].get()) if 'source' in self.extra_inputs else 0
                    sink = int(self.extra_inputs['sink'].get()) if 'sink' in self.extra_inputs else len(adj)-1
                    result = ford_fulkerson(adj, source, sink)
                else:
                    result = "Unknown adjacency algorithm."
            else:
                result = "Unknown input type."
        except Exception as e:
            result = f"Error: {e}"

        self.result_label.config(
            text=result
        )
