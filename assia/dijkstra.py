import heapq

class DijkstraSolver:
    def __init__(self):
        pass
    
    def dijkstra(self, graph, start):
        n = len(graph)
        dist = [float('infinity')] * n
        dist[start] = 0
        heap = [(0, start)]
        
        while heap:
            current_dist, u = heapq.heappop(heap)
            
            if current_dist > dist[u]:
                continue
                
            for v in range(n):
                if graph[u][v] > 0:  # Only if there's an edge
                    distance = current_dist + graph[u][v]
                    if distance < dist[v]:
                        dist[v] = distance
                        heapq.heappush(heap, (distance, v))
        
        return dist