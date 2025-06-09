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