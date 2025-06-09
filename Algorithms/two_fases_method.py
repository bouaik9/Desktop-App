
import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
from collections import deque
import numpy as np
def two_phase_method(matrix):
    """
    Two-phase simplex for maximization in standard form.

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

    # Phase 1: add artificial variables for constraints with b < 0
    A1 = np.copy(A)
    b1 = np.copy(b)
    art_vars = []
    for i in range(len(b1)):
        if b1[i] < 0:
            A1[i, :] *= -1
            b1[i] *= -1
        art = np.zeros((len(b1), 1))
        art[i, 0] = 1
        A1 = np.hstack([A1, art])
        art_vars.append(A1.shape[1] - 1)
    c1 = np.zeros(A1.shape[1])
    for idx in art_vars:
        c1[idx] = 1

    # Solve phase 1
    tableau = np.zeros((len(b1) + 1, A1.shape[1] + 1))
    tableau[:-1, :-1] = A1
    tableau[:-1, -1] = b1
    tableau[-1, :-1] = c1
    tableau[-1, -1] = 0

    basis = art_vars.copy()
    def pivot(tableau, row, col):
        tableau[row, :] /= tableau[row, col]
        for r in range(tableau.shape[0]):
            if r != row:
                tableau[r, :] -= tableau[r, col] * tableau[row, :]

    # Phase 1 simplex
    while True:
        col = next((j for j in range(tableau.shape[1] - 1) if tableau[-1, j] > 1e-8), None)
        if col is None:
            break
        ratios = []
        for i in range(len(b1)):
            if tableau[i, col] > 1e-8:
                ratios.append((tableau[i, -1] / tableau[i, col], i))
        if not ratios:
            return "Infeasible problem."
        _, row = min(ratios)
        pivot(tableau, row, col)
        basis[row] = col

    if abs(tableau[-1, -1]) > 1e-8:
        return "Infeasible problem (artificial variables remain)."

    # Remove artificial variables, proceed to phase 2
    keep_cols = [j for j in range(A1.shape[1]) if j not in art_vars]
    A2 = A1[:, keep_cols]
    tableau2 = np.zeros((len(b1) + 1, len(keep_cols) + 1))
    tableau2[:-1, :-1] = A2
    tableau2[:-1, -1] = b1
    tableau2[-1, :-1] = -c
    tableau2[-1, -1] = 0

    # Find new basis indices
    basis = []
    for i in range(len(b1)):
        row = tableau2[i, :-1]
        idx = np.where(np.abs(row - 1) < 1e-8)[0]
        if len(idx) == 1 and np.all(np.abs(np.delete(row, idx[0])) < 1e-8):
            basis.append(idx[0])
        else:
            basis.append(0)  # fallback

    # Phase 2 simplex
    while True:
        col = next((j for j in range(tableau2.shape[1] - 1) if tableau2[-1, j] < -1e-8), None)
        if col is None:
            break
        ratios = []
        for i in range(len(b1)):
            if tableau2[i, col] > 1e-8:
                ratios.append((tableau2[i, -1] / tableau2[i, col], i))
        if not ratios:
            return "Unbounded solution."
        _, row = min(ratios)
        pivot(tableau2, row, col)
        basis[row] = col

    solution = np.zeros(len(keep_cols))
    for i, bi in enumerate(basis):
        solution[bi] = tableau2[i, -1]
    obj = tableau2[-1, -1]
    return f"Solution optimale : x = {solution.tolist()}\nValeur de la fonction objectif : {obj}"
