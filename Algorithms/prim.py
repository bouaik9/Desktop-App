
from utilities.graph_from_matrix import draw_graph_from_matrix
import threading
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
    viz_thread = threading.Thread(target=draw_graph_from_matrix, args=(adj, edges), kwargs={"path_color": "red"})
    viz_thread.start()

    return f"Arêtes de l'arbre : {edges}\nPoids total : {total}"