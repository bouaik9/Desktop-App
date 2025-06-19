import threading

from utilities.graph_coloring_draw import graph_coloring_draw
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


