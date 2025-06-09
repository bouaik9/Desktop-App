import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_graph_from_matrix(matrix_str, edges=None, path_color='red', colors=None):
    """
    Draws a graph from an adjacency matrix string using NetworkX and Matplotlib,
    and embeds it in a Tkinter window. Optionally colors a specified path and nodes.

    Args:
        matrix_str (list of lists): The adjacency matrix as a 2D list (numbers).
        edges (list, optional): A list of edge tuples representing the path to color.
        path_color (str, optional): The color to use for the path.
        colors (dict, optional): A dict mapping node -> color. E.g. {0: 'red', 1: 'blue'}
    """
    try:
        matrix = matrix_str

        graph = nx.Graph()  # Undirected graph
        num_nodes = len(matrix)
        graph.add_nodes_from(range(num_nodes))

        for i in edges:
            graph.add_edge(i[0], i[1], weight=i[2])  # Add edges for the path

    

        window = tk.Tk()
        window.title("Graph from Adjacency Matrix")

        figure, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#23272e")

        # Edge colors, color path edges differently if edges list provided
  

        # Node colors - from dictionary or default color
        if colors:
            # Make a list of colors corresponding to node order
            node_colors = [colors.get(node, "#4f8cff") for node in graph.nodes()]
        else:
            node_colors = ["#4f8cff"] * num_nodes

        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, ax=ax, with_labels=True, node_color=node_colors,
                node_size=800, font_size=12, font_color="white",
                font_weight="bold", edge_color="red")

        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=edge_labels,
                                    font_size=10, font_color="blue")

        figure.patch.set_facecolor("#23272e")

        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        window.mainloop()

    except Exception as e:
        print(f"Error: {e}")
        tk.messagebox.showerror("Error", str(e))


if __name__ == '__main__':
    adjacency_matrix_str = [
        [0, 2, 4, 0],
        [2, 0, 1, 0],
        [4, 1, 0, 7],
        [0, 0, 7, 0]
    ]

    path = [(0, 1), (1, 2), (2, 3)]
    node_colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}  # color dict by node

    draw_graph_from_matrix(adjacency_matrix_str, edges=path, path_color='red', colors=node_colors)
