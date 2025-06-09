import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
from collections import deque
import numpy as np
def ford_fulkerson(adj, source=0, sink=None):
    
    n = len(adj)
    if sink is None:
        sink = n - 1
    residual = [row[:] for row in adj]
    max_flow = 0
    while True:
        parent = [-1] * n
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in range(n):
                if residual[u][v] > 0 and parent[v] == -1 and v != source:
                    parent[v] = u
                    queue.append(v)
        if parent[sink] == -1:
            break
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, residual[parent[s]][s])
            s = parent[s]
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u
    return f"Ford-Fulkerson max flow from {source} to {sink}: {max_flow}"