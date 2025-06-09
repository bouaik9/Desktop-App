import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
from collections import deque
import numpy as np

from utilities.graph_from_matrix import draw_graph_from_matrix
def kruskal(adj):
    n = len(adj)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append((adj[i][j], i, j))
    edges.sort()
    parent = list(range(n))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    mst = []
    total = 0
    for w, u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            mst.append((u, v))
            total += w
            parent[ru] = rv

    draw_graph_from_matrix(adj, mst, path_color="red")
    return f"Kruskal MST edges: {[(u, v, adj[u][v]) for u, v in mst]}\nTotal weight: {total}"