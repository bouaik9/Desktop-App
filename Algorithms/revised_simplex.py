import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
from collections import deque
import numpy as np
def revised_simplex_method(matrix):
    """
    Simplified revised simplex for maximization in standard form.

    Example input matrix:
    [
        [1, 1, 1, 40],
        [2, 1, 0, 60],
        [0, 1, 1, 50],
        [3, 2, 1, 0]
    ]
    (Last row is the objective coefficients, last column is the RHS.)
    """

    mat = np.array(matrix, dtype=float)
    m, n = mat.shape
    num_vars = n - 1
    num_constraints = m - 1

    A = mat[:-1, :-1]
    b = mat[:-1, -1]
    c = mat[-1, :-1]

    # Initial basis: last num_constraints variables (assume slack variables)
    basis = list(range(num_vars - num_constraints, num_vars))
    N = [j for j in range(num_vars) if j not in basis]

    B = A[:, basis]
    xB = np.linalg.solve(B, b)
    iter_count = 0
    while True:
        iter_count += 1
        B = A[:, basis]
        cB = c[basis]
        try:
            y = np.linalg.solve(B.T, cB)
        except np.linalg.LinAlgError:
            return "Problème numérique : base singulière."
        reduced_costs = []
        for j in N:
            aj = A[:, j]
            rc = c[j] - y @ aj
            reduced_costs.append((rc, j))
        entering = next(((rc, j) for rc, j in reduced_costs if rc > 1e-8), None)
        if not entering:
            # Optimal
            x = np.zeros(num_vars)
            x[basis] = xB
            obj = c @ x
            return f"Solution optimale : x = {x.tolist()}\nValeur de la fonction objectif : {obj}\nItérations : {iter_count}"
        _, j = max(reduced_costs, key=lambda x: x[0])
        d = np.linalg.solve(B, A[:, j])
        if all(d <= 1e-8):
            return "Solution non bornée."
        ratios = [(xB[i] / d[i], i) for i in range(len(d)) if d[i] > 1e-8]
        theta, leaving_idx = min(ratios)
        basis[leaving_idx] = j
        N = [k for k in range(num_vars) if k not in basis]
        xB = xB - theta * d
        xB[leaving_idx] = theta
