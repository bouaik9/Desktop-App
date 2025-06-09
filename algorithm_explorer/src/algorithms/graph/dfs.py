def depth_first_search(graph, start):
    visited = set()
    result = []

    def dfs(node):
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in graph[node]:
                dfs(neighbor)

    dfs(start)
    return result

# Example usage:
# graph = {
#     'A': ['B', 'C'],
#     'B': ['A', 'D', 'E'],
#     'C': ['A', 'F'],
#     'D': ['B'],
#     'E': ['B', 'F'],
#     'F': ['C', 'E']
# }
# print(depth_first_search(graph, 'A'))  # Output: ['A', 'B', 'D', 'E', 'F', 'C']