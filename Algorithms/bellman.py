def bellman_ford(matrix, start):
    graph = {}
    
    for i in range(len(matrix)):
        graph[i] = {}
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                graph[i][j] = matrix[i][j]
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

bellmans_algorithm = bellman_ford






