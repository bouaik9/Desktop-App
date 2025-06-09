import tkinter as tk
from tkinter import ttk, messagebox, font

# Algorithm metadata
ALGORITHMS = {
    "LP: Revised Simplex Method": {
        "description": "An efficient version of the simplex method for solving linear programming problems.",
        "input_type": "matrix"
    },
    "Depth-First Search (DFS)": {
        "description": "A graph traversal algorithm that explores as far as possible along each branch before backtracking.",
        "input_type": "adjacency"
    },
    "Bellman’s Algorithm": {
        "description": "Finds shortest paths from a single source vertex to all other vertices in a weighted digraph.",
        "input_type": "adjacency"
    },
    "Prim’s Algorithm": {
        "description": "Finds a minimum spanning tree for a weighted undirected graph.",
        "input_type": "adjacency"
    },
    "LP: Simplex Algorithm": {
        "description": "A popular algorithm for linear programming problems.",
        "input_type": "matrix"
    },
    "Breadth-First Search (BFS)": {
        "description": "A graph traversal algorithm that explores all neighbors at the present depth before moving on.",
        "input_type": "adjacency"
    },
    "Dijkstra’s Algorithm": {
        "description": "Finds the shortest path between nodes in a graph.",
        "input_type": "adjacency"
    },
    "Kruskal’s Algorithm": {
        "description": "Finds a minimum spanning tree for a connected weighted graph.",
        "input_type": "adjacency"
    },
    "LP: Two-Phase Method": {
        "description": "Solves linear programming problems with artificial variables.",
        "input_type": "matrix"
    },
    "Graph Coloring": {
        "description": "Assigns colors to graph vertices so that no two adjacent vertices share the same color.",
        "input_type": "adjacency"
    },
    "Bellman-Ford Algorithm": {
        "description": "Computes shortest paths from a single source vertex to all other vertices in a weighted digraph.",
        "input_type": "adjacency"
    },
    "Ford-Fulkerson Algorithm": {
        "description": "Computes the maximum flow in a flow network.",
        "input_type": "adjacency"
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
    # matrix: [A|b|c] where last row is objective coefficients, last column is b
    # This is a stub for demonstration; a full simplex implementation is complex.
    return "Simplex Algorithm: [Demo] Solution not implemented."

def revised_simplex_method(matrix):
    return "Revised Simplex Method: [Demo] Solution not implemented."

def two_phase_method(matrix):
    return "Two-Phase Method: [Demo] Solution not implemented."

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
    n = len(adj)
    dist = [float('inf')] * n
    dist[start] = 0
    for _ in range(n-1):
        for u in range(n):
            for v, w in enumerate(adj[u]):
                if w and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    # Check for negative cycles
    for u in range(n):
        for v, w in enumerate(adj[u]):
            if w and dist[u] + w < dist[v]:
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
            edges.append((u, v, adj[u][v]))
            total += adj[u][v]
            selected[v] = True
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
            mst.append((u, v, w))
            total += w
            parent[ru] = rv
    return f"Kruskal MST edges: {mst}\nTotal weight: {total}"

def graph_coloring(adj):
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
    return f"Vertex colors: {result}\nNumber of colors used: {max(result)+1}"

def ford_fulkerson(adj, source=0, sink=None):
    from collections import deque
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
        self.selected_algorithm = None

        self.container = tk.Frame(self, bg=BG_COLOR)
        self.container.pack(fill="both", expand=True)

        self.show_selection_view()

    def show_selection_view(self):
        for widget in self.container.winfo_children():
            widget.destroy()
        AlgorithmSelectionView(self.container, self).pack(fill="both", expand=True)

    def show_input_view(self, algorithm_name):
        self.selected_algorithm = algorithm_name
        for widget in self.container.winfo_children():
            widget.destroy()
        AlgorithmInputView(self.container, self, algorithm_name).pack(fill="both", expand=True)

class AlgorithmSelectionView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_COLOR)
        self.controller = controller

        title_font = font.Font(family="Segoe UI", size=22, weight="bold")
        label = tk.Label(self, text="Choose an Algorithm", font=title_font,
                         bg=BG_COLOR, fg=ACCENT_COLOR, pady=20)
        label.pack()

        button_frame = tk.Frame(self, bg=BG_COLOR)
        button_frame.pack(pady=10)

        btn_font = font.Font(family="Segoe UI", size=12)
        for algo in ALGORITHMS:
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

class AlgorithmInputView(tk.Frame):
    def __init__(self, parent, controller, algorithm_name):
        super().__init__(parent, bg=BG_COLOR)
        self.controller = controller
        self.algorithm_name = algorithm_name

        title_font = font.Font(family="Segoe UI", size=18, weight="bold")
        label = tk.Label(self, text=algorithm_name, font=title_font,
                         bg=BG_COLOR, fg=ACCENT_COLOR, pady=16)
        label.pack()

        # Input area
        input_label = tk.Label(self, text="Enter input data below:",
                               font=("Segoe UI", 12), bg=BG_COLOR, fg=FG_COLOR)
        input_label.pack(anchor="w", padx=32)

        self.extra_inputs = {}

        # Adjacency matrix input
        self.input_text = tk.Text(self, height=8, width=48, font=("Consolas", 12),
                                  bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, bd=1, relief="solid")
        self.input_text.pack(padx=32, pady=8)

        # Additional fields for certain algorithms
        algo = self.algorithm_name
        if algo in ["Dijkstra’s Algorithm", "Bellman-Ford Algorithm", "Bellman’s Algorithm"]:
            # Start node input
            start_frame = tk.Frame(self, bg=BG_COLOR)
            start_frame.pack(anchor="w", padx=32, pady=(0, 8))
            tk.Label(start_frame, text="Start node:", font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
            self.extra_inputs['start'] = tk.Entry(start_frame, width=5, font=("Consolas", 12), bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
            self.extra_inputs['start'].pack(side="left", padx=(8, 0))
            self.extra_inputs['start'].insert(0, "0")
        elif algo == "Ford-Fulkerson Algorithm":
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
        elif algo in ["LP: Simplex Algorithm", "LP: Revised Simplex Method", "LP: Two-Phase Method"]:
            # Show a hint for matrix input
            hint = (
                "Matrix input format:\n"
                "Each row = constraint or objective, values separated by space/comma.\n"
                "Last row = objective coefficients.\n"
                "Last column = right-hand side (b).\n"
                "Example:\n"
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
            btn_frame, text="Back",
            font=("Segoe UI", 11),
            bg=BUTTON_BG, fg=BUTTON_FG,
            activebackground=ACCENT_COLOR, activeforeground=FG_COLOR,
            relief="flat", padx=10, pady=6,
            command=controller.show_selection_view
        )
        back_btn.grid(row=0, column=2, padx=8)

        # Result label
        self.result_label = tk.Label(self, text="", font=("Segoe UI", 12, "italic"),
                                    bg=BG_COLOR, fg=ACCENT_COLOR, wraplength=400, pady=10)
        self.result_label.pack()

    def show_documentation(self):
        desc = ALGORITHMS[self.algorithm_name]["description"]
        messagebox.showinfo(f"{self.algorithm_name} - Documentation", desc)

    def process_input(self):
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("Input Required", "Please enter input data for the algorithm.")
            return

        input_type = ALGORITHMS[self.algorithm_name]["input_type"]
        try:
            if input_type == "matrix":
                matrix = parse_matrix_input(user_input)
                if self.algorithm_name == "LP: Simplex Algorithm":
                    result = simplex_algorithm(matrix)
                elif self.algorithm_name == "LP: Revised Simplex Method":
                    result = revised_simplex_method(matrix)
                elif self.algorithm_name == "LP: Two-Phase Method":
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
                elif algo == "Dijkstra’s Algorithm":
                    start = int(self.extra_inputs['start'].get()) if 'start' in self.extra_inputs else 0
                    result = dijkstra(adj, start)
                elif algo == "Bellman-Ford Algorithm":
                    start = int(self.extra_inputs['start'].get()) if 'start' in self.extra_inputs else 0
                    result = bellman_ford(adj, start)
                elif algo == "Bellman’s Algorithm":
                    start = int(self.extra_inputs['start'].get()) if 'start' in self.extra_inputs else 0
                    result = bellmans_algorithm(adj, start)
                elif algo == "Prim’s Algorithm":
                    result = prim(adj)
                elif algo == "Kruskal’s Algorithm":
                    result = kruskal(adj)
                elif algo == "Graph Coloring":
                    result = graph_coloring(adj)
                elif algo == "Ford-Fulkerson Algorithm":
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
            text=f"Result for '{self.algorithm_name}':\n\n{result}"
        )

if __name__ == "__main__":
    app = App()
    app.mainloop()