def bellman_ford(graph, start):
    distance = {vertex: float('inf') for vertex in graph}
    distance[start] = 0

    for _ in range(len(graph) - 1):
        for vertex in graph:
            for neighbor, weight in graph[vertex].items():
                if distance[vertex] + weight < distance[neighbor]:
                    distance[neighbor] = distance[vertex] + weight

    for vertex in graph:
        for neighbor, weight in graph[vertex].items():
            if distance[vertex] + weight < distance[neighbor]:
                raise ValueError("Graph contains a negative-weight cycle")

    return distance

# Example usage:
# graph = {
#     'A': {'B': 1, 'C': 4},
#     'B': {'C': 2, 'D': 5},
#     'C': {'D': 1},
#     'D': {}
# }
# print(bellman_ford(graph, 'A'))  # Output: {'A': 0, 'B': 1, 'C': 3, 'D': 4}