import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
from collections import deque
import numpy as np
import numpy as np
import numpy as np
from graph_coloring_draw import graph_coloring_draw
from graph_from_matrix import draw_graph_from_matrix

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

# --- Linear Programming Algorithms (Simplex, Revised Simplex, Two-Phase) ---
# For demonstration, these are simplified and only handle maximization problems in standard form.

def simplex_algorithm(matrix):
    """
    Naive simplex implementation for maximization problems in standard form.
    Assumes matrix is [A|b] with last row as objective coefficients, last column as b.

    Example input matrix:
    [
        [1, 1, 1, 40],
        [2, 1, 0, 60],
        [0, 1, 1, 50],
        [3, 2, 1, 0]
    ]
    (Last row is the objective coefficients, last column is the RHS.)
    """

    mat = np.array(matrix, dtype=float)
    m, n = mat.shape
    num_vars = n - 1
    num_constraints = m - 1

    # Split into A, b, c
    A = mat[:-1, :-1]
    b = mat[:-1, -1]
    c = mat[-1, :-1]
    tableau = np.zeros((m, n))
    tableau[:-1, :] = mat[:-1, :]
    tableau[-1, :-1] = -c
    tableau[-1, -1] = 0

    basis = list(range(num_vars - num_constraints, num_vars))

    def pivot(tableau, row, col):
        tableau[row, :] /= tableau[row, col]
        for r in range(tableau.shape[0]):
            if r != row:
                tableau[r, :] -= tableau[r, col] * tableau[row, :]

    while True:
        # Bland's rule: choose smallest index with negative cost
        col = next((j for j in range(n - 1) if tableau[-1, j] < -1e-8), None)
        if col is None:
            break  # optimal
        # Minimum ratio test
        ratios = []
        for i in range(m - 1):
            if tableau[i, col] > 1e-8:
                ratios.append((tableau[i, -1] / tableau[i, col], i))
        if not ratios:
            return "Unbounded solution."
        _, row = min(ratios)
        pivot(tableau, row, col)
        basis[row] = col

    solution = np.zeros(n - 1)
    for i, bi in enumerate(basis):
        solution[bi] = tableau[i, -1]
    obj = tableau[-1, -1]
    return f"Solution optimale : x = {solution.tolist()}\nValeur de la fonction objectif : {obj}"

def revised_simplex_method(matrix):
    """
    Simplified revised simplex for maximization in standard form.

    Example input matrix:
    [
        [1, 1, 1, 40],
        [2, 1, 0, 60],
        [0, 1, 1, 50],
        [3, 2, 1, 0]
    ]
    (Last row is the objective coefficients, last column is the RHS.)
    """

    mat = np.array(matrix, dtype=float)
    m, n = mat.shape
    num_vars = n - 1
    num_constraints = m - 1

    A = mat[:-1, :-1]
    b = mat[:-1, -1]
    c = mat[-1, :-1]

    # Initial basis: last num_constraints variables (assume slack variables)
    basis = list(range(num_vars - num_constraints, num_vars))
    N = [j for j in range(num_vars) if j not in basis]

    B = A[:, basis]
    xB = np.linalg.solve(B, b)
    iter_count = 0
    while True:
        iter_count += 1
        B = A[:, basis]
        cB = c[basis]
        try:
            y = np.linalg.solve(B.T, cB)
        except np.linalg.LinAlgError:
            return "Numerical issue: singular basis."
        reduced_costs = []
        for j in N:
            aj = A[:, j]
            rc = c[j] - y @ aj
            reduced_costs.append((rc, j))
        entering = next(((rc, j) for rc, j in reduced_costs if rc > 1e-8), None)
        if not entering:
            # Optimal
            x = np.zeros(num_vars)
            x[basis] = xB
            obj = c @ x
            return f"Solution optimale : x = {x.tolist()}\nValeur de la fonction objectif : {obj}\nItérations : {iter_count}"
        _, j = max(reduced_costs, key=lambda x: x[0])
        d = np.linalg.solve(B, A[:, j])
        if all(d <= 1e-8):
            return "Unbounded solution."
        ratios = [(xB[i] / d[i], i) for i in range(len(d)) if d[i] > 1e-8]
        theta, leaving_idx = min(ratios)
        basis[leaving_idx] = j
        N = [k for k in range(num_vars) if k not in basis]
        xB = xB - theta * d
        xB[leaving_idx] = theta

def two_phase_method(matrix):
    """
    Two-phase simplex for maximization in standard form.

    Example input matrix:
    [
        [1, 1, 1, 40],
        [2, 1, 0, 60],
        [0, 1, 1, 50],
        [3, 2, 1, 0]
    ]
    (Last row is the objective coefficients, last column is the RHS.)
    """

    mat = np.array(matrix, dtype=float)
    m, n = mat.shape
    num_vars = n - 1
    num_constraints = m - 1

    A = mat[:-1, :-1]
    b = mat[:-1, -1]
    c = mat[-1, :-1]

    # Phase 1: add artificial variables for constraints with b < 0
    A1 = np.copy(A)
    b1 = np.copy(b)
    art_vars = []
    for i in range(len(b1)):
        if b1[i] < 0:
            A1[i, :] *= -1
            b1[i] *= -1
        art = np.zeros((len(b1), 1))
        art[i, 0] = 1
        A1 = np.hstack([A1, art])
        art_vars.append(A1.shape[1] - 1)
    c1 = np.zeros(A1.shape[1])
    for idx in art_vars:
        c1[idx] = 1

    # Solve phase 1
    tableau = np.zeros((len(b1) + 1, A1.shape[1] + 1))
    tableau[:-1, :-1] = A1
    tableau[:-1, -1] = b1
    tableau[-1, :-1] = c1
    tableau[-1, -1] = 0

    basis = art_vars.copy()
    def pivot(tableau, row, col):
        tableau[row, :] /= tableau[row, col]
        for r in range(tableau.shape[0]):
            if r != row:
                tableau[r, :] -= tableau[r, col] * tableau[row, :]

    # Phase 1 simplex
    while True:
        col = next((j for j in range(tableau.shape[1] - 1) if tableau[-1, j] > 1e-8), None)
        if col is None:
            break
        ratios = []
        for i in range(len(b1)):
            if tableau[i, col] > 1e-8:
                ratios.append((tableau[i, -1] / tableau[i, col], i))
        if not ratios:
            return "Infeasible problem."
        _, row = min(ratios)
        pivot(tableau, row, col)
        basis[row] = col

    if abs(tableau[-1, -1]) > 1e-8:
        return "Infeasible problem (artificial variables remain)."

    # Remove artificial variables, proceed to phase 2
    keep_cols = [j for j in range(A1.shape[1]) if j not in art_vars]
    A2 = A1[:, keep_cols]
    tableau2 = np.zeros((len(b1) + 1, len(keep_cols) + 1))
    tableau2[:-1, :-1] = A2
    tableau2[:-1, -1] = b1
    tableau2[-1, :-1] = -c
    tableau2[-1, -1] = 0

    # Find new basis indices
    basis = []
    for i in range(len(b1)):
        row = tableau2[i, :-1]
        idx = np.where(np.abs(row - 1) < 1e-8)[0]
        if len(idx) == 1 and np.all(np.abs(np.delete(row, idx[0])) < 1e-8):
            basis.append(idx[0])
        else:
            basis.append(0)  # fallback

    # Phase 2 simplex
    while True:
        col = next((j for j in range(tableau2.shape[1] - 1) if tableau2[-1, j] < -1e-8), None)
        if col is None:
            break
        ratios = []
        for i in range(len(b1)):
            if tableau2[i, col] > 1e-8:
                ratios.append((tableau2[i, -1] / tableau2[i, col], i))
        if not ratios:
            return "Unbounded solution."
        _, row = min(ratios)
        pivot(tableau2, row, col)
        basis[row] = col

    solution = np.zeros(len(keep_cols))
    for i, bi in enumerate(basis):
        solution[bi] = tableau2[i, -1]
    obj = tableau2[-1, -1]
    return f"Solution optimale : x = {solution.tolist()}\nValeur de la fonction objectif : {obj}"

# --- Graph Algorithms ---

def dfs(adj, start=0):
    visited = []
    def visit(u):
        visited.append(u)
        for v, w in enumerate(adj[u]):
            if w and v not in visited:
                visit(v)
    visit(start)
    return f"DFS order: {visited}"

def bfs(adj, start=0):
    from collections import deque
    visited = [start]
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v, w in enumerate(adj[u]):
            if w and v not in visited:
                visited.append(v)
                queue.append(v)
    return f"BFS order: {visited}"

def dijkstra(adj, start=0):
    import heapq
    n = len(adj)
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in enumerate(adj[u]):
            if w and dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
            
    return f"Dijkstra distances from {start}: {dist}"

def bellman_ford(adj, start=0):
    import threading

    n = len(adj)
    dist = [float('inf')] * n
    dist[start] = 0

    for _ in range(n - 1):
        for u in range(n):
            for v, w in enumerate(adj[u]):
                if w != 0 and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

    # Check for negative cycles
    for u in range(n):
        for v, w in enumerate(adj[u]):
            if w != 0 and dist[u] + w < dist[v]:
                return "Negative cycle detected."



    return f"Bellman-Ford distances from {start}: {dist}"


def bellmans_algorithm(adj, start=0):
    # Alias for Bellman-Ford
    return bellman_ford(adj, start)

def prim(adj):
    n = len(adj)
    selected = [False] * n
    selected[0] = True
    edges = []
    total = 0
    for _ in range(n-1):
        min_w = float('inf')
        u = v = -1
        for i in range(n):
            if selected[i]:
                for j in range(n):
                    if not selected[j] and adj[i][j] and adj[i][j] < min_w:
                        min_w = adj[i][j]
                        u, v = i, j
        if u != -1 and v != -1:
            edges.append((int(u), int(v), adj[u][v]))
            total += adj[u][v]
            selected[v] = True

    # Launch visualization in background
    draw_graph_from_matrix(adj, edges, path_color="red")

    return f"Prim MST edges: {edges}\nTotal weight: {total}"

def kruskal(adj):
    n = len(adj)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append((adj[i][j], i, j))
    edges.sort()
    parent = list(range(n))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    mst = []
    total = 0
    for w, u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            mst.append((u, v))
            total += w
            parent[ru] = rv

    draw_graph_from_matrix(adj, mst, path_color="red")
    return f"Kruskal MST edges: {[(u, v, adj[u][v]) for u, v in mst]}\nTotal weight: {total}"

import threading
import webbrowser

def graph_coloring(adj):
    color_names = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'cyan']
    n = len(adj)
    result = [-1] * n
    result[0] = 0

    for u in range(1, n):
        used = set()
        for v in range(n):
            if adj[u][v] and result[v] != -1:
                used.add(result[v])
        color = 0
        while color in used:
            color += 1
        result[u] = color

    # Map node indices to color names
    node_colors = {i: color_names[result[i]] for i in range(n)}

    # Run draw_graph_from_matrix in a separate thread
    def draw():
        graph_coloring_draw(adj, colors=node_colors)

    threading.Thread(target=draw, daemon=True).start()

    return f"Nombre de couleurs utilisées : {max(result)+1}"


def ford_fulkerson(adj, source=0, sink=None):
    
    n = len(adj)
    if sink is None:
        sink = n - 1
    residual = [row[:] for row in adj]
    max_flow = 0
    while True:
        parent = [-1] * n
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in range(n):
                if residual[u][v] > 0 and parent[v] == -1 and v != source:
                    parent[v] = u
                    queue.append(v)
        if parent[sink] == -1:
            break
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, residual[parent[s]][s])
            s = parent[s]
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u
    return f"Ford-Fulkerson max flow from {source} to {sink}: {max_flow}"

# --- Main App and Views ---

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

if __name__ == "__main__":
    app = App()
    app.mainloop()