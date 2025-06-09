import tkinter as tk
from tkinter import ttk, messagebox
from simplexe import SimplexeSolver
from bfs import BFSTraversal
from dijkstra import DijkstraSolver
from kruskal import KruskalSolver
import numpy as np
from collections import defaultdict

class AlgorithmApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Algorithms Visualizer")
        self.geometry("1000x700")
        self.configure(bg="#f5f5f5")
        
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f5f5f5")
        self.style.configure("TLabel", background="#f5f5f5", font=('Arial', 10))
        self.style.configure("TButton", font=('Arial', 10), padding=5)
        self.style.configure("Header.TLabel", font=('Arial', 14, 'bold'))
        
        self.create_widgets()
        self.current_algo = None
    
    def create_widgets(self):
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(pady=10, fill="x")
        
        ttk.Label(header_frame, text="Algorithm Visualizer", style="Header.TLabel").pack()
        
        # Algorithm Selection
        algo_frame = ttk.Frame(self)
        algo_frame.pack(pady=10, fill="x")
        
        ttk.Label(algo_frame, text="Select Algorithm:").pack(side="left", padx=5)
        
        self.algo_var = tk.StringVar()
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algo_var,
                                values=["Simplex", "BFS", "Dijkstra", "Kruskal"])
        algo_combo.pack(side="left", padx=5)
        algo_combo.bind("<<ComboboxSelected>>", self.on_algo_select)
        algo_combo.current(0)
        
        # Input Frame
        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(pady=10, fill="both", expand=True)
        
        # Result Area
        result_frame = ttk.Frame(self)
        result_frame.pack(pady=10, fill="both", expand=True)
        
        ttk.Label(result_frame, text="Results:").pack(anchor="w")
        
        self.result_text = tk.Text(result_frame, height=15, width=100, state="disabled",
                                 font=('Courier New', 10), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.pack(fill="both", expand=True)
        
        # Initialize first algorithm
        self.on_algo_select()
    
    def on_algo_select(self, event=None):
        algo = self.algo_var.get()
        
        # Clear previous widgets
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        
        if algo == "Simplex":
            self.setup_simplex_inputs()
        elif algo == "BFS":
            self.setup_bfs_inputs()
        elif algo == "Dijkstra":
            self.setup_dijkstra_inputs()
        elif algo == "Kruskal":
            self.setup_kruskal_inputs()
    
    def setup_simplex_inputs(self):
        # Variables input
        var_frame = ttk.Frame(self.input_frame)
        var_frame.pack(pady=5)
        
        ttk.Label(var_frame, text="Number of Variables:").grid(row=0, column=0, padx=5)
        self.var_count = tk.Spinbox(var_frame, from_=1, to=10, width=5)
        self.var_count.grid(row=0, column=1, padx=5)
        
        # Constraints input
        constr_frame = ttk.Frame(self.input_frame)
        constr_frame.pack(pady=5)
        
        ttk.Label(constr_frame, text="Number of Constraints:").grid(row=0, column=0, padx=5)
        self.constr_count = tk.Spinbox(constr_frame, from_=1, to=10, width=5)
        self.constr_count.grid(row=0, column=1, padx=5)
        
        # Generate button
        gen_button = ttk.Button(self.input_frame, text="Generate Tableau",
                              command=self.generate_simplex_table)
        gen_button.pack(pady=10)
        
        # Table frame (will be populated later)
        self.table_frame = ttk.Frame(self.input_frame)
        self.table_frame.pack(fill="both", expand=True)
        
        # Solve button (initially disabled)
        self.solve_button = ttk.Button(self.input_frame, text="Solve",
                                      state="disabled", command=self.solve_simplex)
        self.solve_button.pack(pady=10)
    
    def generate_simplex_table(self):
        try:
            n_vars = int(self.var_count.get())
            n_constr = int(self.constr_count.get())
            
            # Clear previous table
            for widget in self.table_frame.winfo_children():
                widget.destroy()
            
            # Create labels for headers
            headers = [f"x{i+1}" for i in range(n_vars)] + ["RHS"]
            for j, header in enumerate(headers):
                ttk.Label(self.table_frame, text=header).grid(row=0, column=j+1, padx=5, pady=2)
            
            # Create constraint rows
            self.simplex_entries = []
            for i in range(n_constr):
                ttk.Label(self.table_frame, text=f"Constraint {i+1}").grid(row=i+1, column=0, padx=5)
                row_entries = []
                for j in range(n_vars + 1):  # +1 for RHS
                    entry = ttk.Entry(self.table_frame, width=8)
                    entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                    row_entries.append(entry)
                self.simplex_entries.append(row_entries)
            
            # Objective function row
            ttk.Label(self.table_frame, text="Objective").grid(row=n_constr+1, column=0, padx=5)
            obj_entries = []
            for j in range(n_vars):
                entry = ttk.Entry(self.table_frame, width=8)
                entry.grid(row=n_constr+1, column=j+1, padx=2, pady=2)
                obj_entries.append(entry)
            self.simplex_entries.append(obj_entries)
            
            # Optimization type
            self.opt_type = tk.StringVar(value="max")
            ttk.Radiobutton(self.table_frame, text="Maximize", variable=self.opt_type,
                           value="max").grid(row=n_constr+2, column=1, columnspan=2)
            ttk.Radiobutton(self.table_frame, text="Minimize", variable=self.opt_type,
                           value="min").grid(row=n_constr+2, column=3, columnspan=2)
            
            self.solve_button.config(state="normal")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
    
    def solve_simplex(self):
        try:
            n_vars = int(self.var_count.get())
            n_constr = int(self.constr_count.get())
            
            # Extract constraints
            constraints = []
            for i in range(n_constr):
                row = []
                for j in range(n_vars + 1):
                    val = float(self.simplex_entries[i][j].get())
                    row.append(val)
                constraints.append(row)
            
            # Extract objective
            objective = []
            for j in range(n_vars):
                val = float(self.simplex_entries[n_constr][j].get())
                objective.append(val)
            
            # Solve
            solver = SimplexeSolver()
            solution = solver.solve(constraints, objective, self.opt_type.get())
            
            # Display results
            self.display_result(solution)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
    
    def setup_bfs_inputs(self):
        # Graph size input
        size_frame = ttk.Frame(self.input_frame)
        size_frame.pack(pady=5)
        
        ttk.Label(size_frame, text="Number of Vertices:").grid(row=0, column=0, padx=5)
        self.bfs_vertex_count = tk.Spinbox(size_frame, from_=1, to=20, width=5)
        self.bfs_vertex_count.grid(row=0, column=1, padx=5)
        
        # Start vertex input
        start_frame = ttk.Frame(self.input_frame)
        start_frame.pack(pady=5)
        
        ttk.Label(start_frame, text="Start Vertex (0-indexed):").grid(row=0, column=0, padx=5)
        self.bfs_start_vertex = tk.Spinbox(start_frame, from_=0, to=19, width=5)
        self.bfs_start_vertex.grid(row=0, column=1, padx=5)
        
        # Generate button
        gen_button = ttk.Button(self.input_frame, text="Generate Adjacency Matrix",
                              command=self.generate_bfs_matrix)
        gen_button.pack(pady=10)
        
        # Matrix frame
        self.bfs_matrix_frame = ttk.Frame(self.input_frame)
        self.bfs_matrix_frame.pack(fill="both", expand=True)
        
        # Solve button
        solve_button = ttk.Button(self.input_frame, text="Run BFS",
                                command=self.solve_bfs)
        solve_button.pack(pady=10)
    
    def generate_bfs_matrix(self):
        try:
            n_vertices = int(self.bfs_vertex_count.get())
            
            # Clear previous matrix
            for widget in self.bfs_matrix_frame.winfo_children():
                widget.destroy()
            
            # Create matrix labels
            for i in range(n_vertices):
                ttk.Label(self.bfs_matrix_frame, text=f"V{i}").grid(row=i+1, column=0, padx=5)
                ttk.Label(self.bfs_matrix_frame, text=f"V{i}").grid(row=0, column=i+1, padx=5)
            
            # Create matrix entries
            self.bfs_matrix_entries = []
            for i in range(n_vertices):
                row_entries = []
                for j in range(n_vertices):
                    entry = ttk.Entry(self.bfs_matrix_frame, width=3)
                    entry.insert(0, "0")
                    entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                    row_entries.append(entry)
                self.bfs_matrix_entries.append(row_entries)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
    
    def solve_bfs(self):
        try:
            n_vertices = int(self.bfs_vertex_count.get())
            start_vertex = int(self.bfs_start_vertex.get())
            
            # Build adjacency list
            adj = [[] for _ in range(n_vertices)]
            for i in range(n_vertices):
                for j in range(n_vertices):
                    if int(self.bfs_matrix_entries[i][j].get()) == 1:
                        adj[i].append(j)
            
            # Run BFS
            solver = BFSTraversal()
            traversal_order = solver.bfs(adj, start_vertex)
            
            # Display results
            result = f"BFS Traversal Order starting from vertex {start_vertex}:\n"
            result += " → ".join(map(str, traversal_order))
            self.display_result(result)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
    
    def setup_dijkstra_inputs(self):
        # Graph size input
        size_frame = ttk.Frame(self.input_frame)
        size_frame.pack(pady=5)
        
        ttk.Label(size_frame, text="Number of Vertices:").grid(row=0, column=0, padx=5)
        self.dij_vertex_count = tk.Spinbox(size_frame, from_=1, to=20, width=5)
        self.dij_vertex_count.grid(row=0, column=1, padx=5)
        
        # Start vertex input
        start_frame = ttk.Frame(self.input_frame)
        start_frame.pack(pady=5)
        
        ttk.Label(start_frame, text="Start Vertex (0-indexed):").grid(row=0, column=0, padx=5)
        self.dij_start_vertex = tk.Spinbox(start_frame, from_=0, to=19, width=5)
        self.dij_start_vertex.grid(row=0, column=1, padx=5)
        
        # Generate button
        gen_button = ttk.Button(self.input_frame, text="Generate Weight Matrix",
                              command=self.generate_dijkstra_matrix)
        gen_button.pack(pady=10)
        
        # Matrix frame
        self.dij_matrix_frame = ttk.Frame(self.input_frame)
        self.dij_matrix_frame.pack(fill="both", expand=True)
        
        # Solve button
        solve_button = ttk.Button(self.input_frame, text="Run Dijkstra",
                                command=self.solve_dijkstra)
        solve_button.pack(pady=10)
    
    def generate_dijkstra_matrix(self):
        try:
            n_vertices = int(self.dij_vertex_count.get())
            
            # Clear previous matrix
            for widget in self.dij_matrix_frame.winfo_children():
                widget.destroy()
            
            # Create matrix labels
            for i in range(n_vertices):
                ttk.Label(self.dij_matrix_frame, text=f"V{i}").grid(row=i+1, column=0, padx=5)
                ttk.Label(self.dij_matrix_frame, text=f"V{i}").grid(row=0, column=i+1, padx=5)
            
            # Create matrix entries
            self.dij_matrix_entries = []
            for i in range(n_vertices):
                row_entries = []
                for j in range(n_vertices):
                    entry = ttk.Entry(self.dij_matrix_frame, width=5)
                    entry.insert(0, "0" if i == j else "∞")
                    entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                    row_entries.append(entry)
                self.dij_matrix_entries.append(row_entries)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
    
    def solve_dijkstra(self):
        try:
            n_vertices = int(self.dij_vertex_count.get())
            start_vertex = int(self.dij_start_vertex.get())
            
            # Build adjacency matrix
            adj_matrix = [[0]*n_vertices for _ in range(n_vertices)]
            for i in range(n_vertices):
                for j in range(n_vertices):
                    val = self.dij_matrix_entries[i][j].get()
                    adj_matrix[i][j] = float('inf') if val == "∞" else int(val)
            
            # Run Dijkstra
            solver = DijkstraSolver()
            distances = solver.dijkstra(adj_matrix, start_vertex)
            
            # Display results
            result = f"Shortest distances from vertex {start_vertex}:\n"
            for i, dist in enumerate(distances):
                result += f"Vertex {i}: {dist if dist != float('inf') else '∞'}\n"
            self.display_result(result)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
    
    def setup_kruskal_inputs(self):
        # Graph size input
        size_frame = ttk.Frame(self.input_frame)
        size_frame.pack(pady=5)
        
        ttk.Label(size_frame, text="Number of Vertices:").grid(row=0, column=0, padx=5)
        self.kruskal_vertex_count = tk.Spinbox(size_frame, from_=1, to=20, width=5)
        self.kruskal_vertex_count.grid(row=0, column=1, padx=5)
        
        ttk.Label(size_frame, text="Number of Edges:").grid(row=1, column=0, padx=5)
        self.kruskal_edge_count = tk.Spinbox(size_frame, from_=1, to=50, width=5)
        self.kruskal_edge_count.grid(row=1, column=1, padx=5)
        
        # Generate button
        gen_button = ttk.Button(self.input_frame, text="Generate Edge List",
                              command=self.generate_kruskal_edges)
        gen_button.pack(pady=10)
        
        # Edges frame
        self.kruskal_edges_frame = ttk.Frame(self.input_frame)
        self.kruskal_edges_frame.pack(fill="both", expand=True)
        
        # Solve button
        solve_button = ttk.Button(self.input_frame, text="Run Kruskal",
                                command=self.solve_kruskal)
        solve_button.pack(pady=10)
    
    def generate_kruskal_edges(self):
        try:
            n_vertices = int(self.kruskal_vertex_count.get())
            n_edges = int(self.kruskal_edge_count.get())
            
            # Clear previous edges
            for widget in self.kruskal_edges_frame.winfo_children():
                widget.destroy()
            
            # Create headers
            headers = ["Edge #", "Vertex u", "Vertex v", "Weight"]
            for col, header in enumerate(headers):
                ttk.Label(self.kruskal_edges_frame, text=header).grid(row=0, column=col, padx=5)
            
            # Create edge entries
            self.kruskal_edge_entries = []
            for i in range(n_edges):
                ttk.Label(self.kruskal_edges_frame, text=str(i+1)).grid(row=i+1, column=0, padx=5)
                
                u_entry = ttk.Entry(self.kruskal_edges_frame, width=5)
                u_entry.grid(row=i+1, column=1, padx=2)
                
                v_entry = ttk.Entry(self.kruskal_edges_frame, width=5)
                v_entry.grid(row=i+1, column=2, padx=2)
                
                w_entry = ttk.Entry(self.kruskal_edges_frame, width=5)
                w_entry.grid(row=i+1, column=3, padx=2)
                
                self.kruskal_edge_entries.append((u_entry, v_entry, w_entry))
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
    
    def solve_kruskal(self):
        try:
            n_vertices = int(self.kruskal_vertex_count.get())
            n_edges = int(self.kruskal_edge_count.get())
            
            # Build edge list
            edges = []
            for u_entry, v_entry, w_entry in self.kruskal_edge_entries:
                u = int(u_entry.get())
                v = int(v_entry.get())
                w = int(w_entry.get())
                edges.append((u, v, w))
            
            # Run Kruskal
            solver = KruskalSolver()
            mst = solver.kruskal(n_vertices, edges)
            
            # Display results
            result = "Minimum Spanning Tree Edges:\n"
            total_weight = 0
            for u, v, w in mst:
                result += f"Edge ({u} - {v}) with weight {w}\n"
                total_weight += w
            result += f"\nTotal MST Weight: {total_weight}"
            self.display_result(result)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
    
    def display_result(self, result):
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, str(result))
        self.result_text.config(state="disabled")

if __name__ == "__main__":
    app = AlgorithmApp()
    app.mainloop()