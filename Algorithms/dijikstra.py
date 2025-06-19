import heapq

def dijkstra(adj, start=0):
    n = len(adj)
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in enumerate(adj[u]):
            if w and dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
            
    return f"Dijkstra distances from {start}: {dist}"