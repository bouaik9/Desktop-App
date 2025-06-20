def dfs(adj, start=0):
    visited = []
    def visit(u):
        visited.append(u)
        for v, w in enumerate(adj[u]):
            if w and v not in visited:
                visit(v)
    visit(start)
    return f"DFS order: {visited}"
