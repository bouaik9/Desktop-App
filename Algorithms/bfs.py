from collections import deque


def bfs(adj, start=0):
    visited = [start]
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v, w in enumerate(adj[u]):
            if w and v not in visited:
                visited.append(v)
                queue.append(v)
    return f"BFS order: {visited}"