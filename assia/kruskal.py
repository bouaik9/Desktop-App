class KruskalSolver:
    def __init__(self):
        pass
    
    def find_parent(self, parent, i):
        if parent[i] == i:
            return i
        return self.find_parent(parent, parent[i])
    
    def kruskal(self, n_vertices, edges):
        result = []
        edges = sorted(edges, key=lambda item: item[2])
        parent = [i for i in range(n_vertices)]
        
        for edge in edges:
            u, v, w = edge
            u_root = self.find_parent(parent, u)
            v_root = self.find_parent(parent, v)
            
            if u_root != v_root:
                result.append(edge)
                parent[v_root] = u_root
        
        return result