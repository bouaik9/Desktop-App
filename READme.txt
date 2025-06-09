
# Graph Algorithm Explorer

A Python desktop application for **visualizing and exploring graph algorithms** and **linear programming methods**. It features an intuitive GUI that allows users to input data and visualize step-by-step solutions.

---

## 🚀 Features

### 🧮 Linear Programming
- Simplex Method  
- Revised Simplex Method  
- Two-Phase Method

### 📈 Graph Theory
- Depth-First Search (DFS)  
- Breadth-First Search (BFS)  
- Dijkstra’s Algorithm  
- Bellman-Ford Algorithm  
- Prim’s Algorithm  
- Kruskal’s Algorithm  
- Graph Coloring  
- Maximum Flow (Ford-Fulkerson)

---

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Desktop App"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

1. **Run the application**
   ```bash
   python main.py
   ```

2. The GUI will launch, providing two main sections:
   - **Linear Programming**
   - **Graph Theory**

3. **Select an algorithm** and input data using the following formats:

### 🔗 Graph Input Format
Enter an **adjacency matrix** (rows separated by newlines; values by space or comma):

```
0 2 4 0
2 0 1 0
0 1 0 7
0 0 7 0
```

### 📊 Linear Programming Format
Enter the **constraint matrix**, where:
- Each row represents a constraint.
- The **last row** is the **objective function**.
- The **last column** is the **right-hand side** (RHS):

```
1 1 1 40
2 1 0 60
0 1 1 50
3 2 1 0
```

---

## 📁 Project Structure

```
.
├── main.py                  # Entry point of the application
├── algorithm_gui.py         # GUI logic
├── Algorithms/              # All implemented algorithms
│   ├── simplex.py
│   ├── revised_simplex.py
│   ├── two_phases_method.py
│   ├── dfs.py
│   ├── bfs.py
│   └── ... (other algorithms)
├── utilities/               # Helper utilities
│   ├── graph_from_matrix.py
│   ├── graph_coloring_draw.py
│   └── minimum_path_from_graph.py
├── requirements.txt
└── README.md
```

---

## 📦 Dependencies

- numpy  
- matplotlib  
- networkx

---

## 🤝 Contributing

Contributions, feature requests, and bug reports are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.
