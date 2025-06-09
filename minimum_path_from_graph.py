import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_graph_from_matrix(matrix_str, path_to_color=None, path_color='red'):
    """
    Draws a graph from an adjacency matrix string using NetworkX and Matplotlib,
    and embeds it in a Tkinter window.  Optionally colors a specified path.

    Args:
        matrix_str (str): A string representation of the adjacency matrix,
                          where rows are separated by newlines and values in each row
                          are separated by spaces or commas.
        path_to_color (list, optional): A list of node indices representing the path to color.
                                         Defaults to None.
        path_color (str, optional): The color to use for the path. Defaults to 'red'.
    """
    try:
        # Parse the adjacency matrix string into a 2D list of floats
        matrix = matrix_str

  

        # Create a directed graph from the adjacency matrix
        graph = nx.Graph()  # Use nx.Graph() for undirected graph
        num_nodes = len(matrix)
        graph.add_nodes_from(range(num_nodes))

        for i in range(num_nodes):
            for j in range(i, num_nodes):  # Add only once for undirected
                if matrix[i][j] != 0:
                    graph.add_edge(i, j, weight=matrix[i][j])


        # Create a Tkinter window
        window = tk.Tk()
        window.title("Graph from Adjacency Matrix")

        # Create a Matplotlib figure and axes
        figure, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#23272e")  # Set background color

        edge_colors = []
        path_edges = set()
        if path_to_color is not None:
            for i in range(len(path_to_color) - 1):
                path_edges.add((path_to_color[i], path_to_color[i + 1]))
                path_edges.add((path_to_color[i + 1], path_to_color[i]))  # For undirected graph

        for edge in graph.edges():
            if edge in path_edges:
                edge_colors.append(path_color)
            else:
                edge_colors.append('black')
        # Draw the graph using NetworkX
        pos = nx.spring_layout(graph)  # You can use other layout algorithms
        nx.draw(graph, pos, ax=ax, with_labels=True, node_color="#4f8cff",
                node_size=800, font_size=12, font_color="white",
                font_weight="bold", edge_color=edge_colors)

        # Add edge weights to the graph
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=edge_labels,
                                    font_size=10, font_color="blue")

        # Color the specified path
      
            
        figure.patch.set_facecolor("#23272e")  # Set figure background color

        # Embed the Matplotlib figure in the Tkinter window
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Update the canvas
        canvas.draw()

        # Run the Tkinter event loop
        window.mainloop()

    except Exception as e:
        print(f"Error: {e}")
        tk.messagebox.showerror("Error", str(e))

if __name__ == '__main__':
    # Example usage:
    adjacency_matrix_str = [
        [0, 2, 4, 0],
        [2, 0, 1, 0],
        [4, 1, 0, 7],
        [0, 0, 7, 0]
    ]
    path = [0, 1, 2, 3]  # Example path: Node 0 -> Node 1 -> Node 2
    draw_graph_from_matrix(adjacency_matrix_str, path_to_color=path, path_color='red')