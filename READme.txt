# Graph Algorithm Explorer

A Python desktop application for visualizing and exploring various graph algorithms and linear programming methods. The application provides an interactive GUI for inputting data and visualizing the results of different algorithms.

## Features

### Linear Programming Algorithms
- Revised Simplex Method
- Simplex Algorithm
- Two-Phase Method

### Graph Theory Algorithms
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- Dijkstra's Algorithm
- Bellman-Ford Algorithm 
- Prim's Algorithm
- Kruskal's Algorithm
- Graph Coloring
- Maximum Flow (Ford-Fulkerson)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Desktop\ App/
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. The GUI will appear with two main categories:
   - Linear Programming
   - Graph Theory

3. Select an algorithm and input your data according to the format:

### For Graph Algorithms
Input an adjacency matrix where each row is space or comma-separated:
```
0 2 4 0
2 0 1 0
0 1 0 7
0 0 7 0
```

### For Linear Programming
Input the constraint matrix with the last row being the objective function and the last column being the RHS:
```
1 1 1 40
2 1 0 60
0 1 1 50
3 2 1 0
```

## Project Structure

- `main.py` - Entry point of the application
- `algorithm_gui.py` - Main GUI implementation
- `Algorithms/` - Implementation of all algorithms
  - `simplex.py`
  - `revised_simplex.py`
  - `two_phases_method.py`
  - `dfs.py`
  - `bfs.py`
  - etc.
- `utilities/` - Helper functions for graph visualization
  - `graph_from_matrix.py`
  - `graph_coloring_draw.py`
  - `minimum_path_from_graph.py`

## Dependencies

- numpy
- matplotlib
- networkx

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.