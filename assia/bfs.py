from collections import deque

class BFSTraversal:
    def __init__(self):
        pass
    
    def bfs(self, adj, start):
        visited = [False] * len(adj)
        queue = deque([start])
        visited[start] = True
        traversal_order = []
        
        while queue:
            vertex = queue.popleft()
            traversal_order.append(vertex)
            
            for neighbor in adj[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        return traversal_order