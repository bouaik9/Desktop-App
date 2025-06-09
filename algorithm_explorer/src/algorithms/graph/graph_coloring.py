def graph_coloring(graph):
    def is_safe(node, color, c):
        for neighbor in range(len(graph)):
            if graph[node][neighbor] == 1 and color[neighbor] == c:
                return False
        return True

    def graph_coloring_util(m, graph, color, node):
        if node == len(graph):
            return True

        for c in range(1, m + 1):
            if is_safe(node, color, c):
                color[node] = c
                if graph_coloring_util(m, graph, color, node + 1):
                    return True
                color[node] = 0

        return False

    m = 3  # Number of colors
    color = [0] * len(graph)

    if graph_coloring_util(m, graph, color, 0):
        return color
    else:
        return None  # No solution found

# Example usage
if __name__ == "__main__":
    graph = [
        [0, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 0]
    ]
    result = graph_coloring(graph)
    print("Coloring of graph:", result)