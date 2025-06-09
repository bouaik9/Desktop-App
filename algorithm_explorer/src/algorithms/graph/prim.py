def prim(graph):
    num_vertices = len(graph)
    selected_nodes = [False] * num_vertices
    num_edges = 0
    selected_nodes[0] = True
    edges = []

    while num_edges < num_vertices - 1:
        minimum = float('inf')
        x = 0
        y = 0

        for i in range(num_vertices):
            if selected_nodes[i]:
                for j in range(num_vertices):
                    if not selected_nodes[j] and graph[i][j]:
                        if minimum > graph[i][j]:
                            minimum = graph[i][j]
                            x = i
                            y = j

        edges.append((x, y, minimum))
        selected_nodes[y] = True
        num_edges += 1

    return edges