import numpy as np

def simplex_algorithm(matrix):
    """
    Naive simplex implementation for maximization problems in standard form.
    Assumes matrix is [A|b] with last row as objective coefficients, last column as b.

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

    # Split into A, b, c
    A = mat[:-1, :-1]
    b = mat[:-1, -1]
    c = mat[-1, :-1]
    tableau = np.zeros((m, n))
    tableau[:-1, :] = mat[:-1, :]
    tableau[-1, :-1] = -c
    tableau[-1, -1] = 0

    basis = list(range(num_vars - num_constraints, num_vars))

    def pivot(tableau, row, col):
        tableau[row, :] /= tableau[row, col]
        for r in range(tableau.shape[0]):
            if r != row:
                tableau[r, :] -= tableau[r, col] * tableau[row, :]

    while True:
        # Bland's rule: choose smallest index with negative cost
        col = next((j for j in range(n - 1) if tableau[-1, j] < -1e-8), None)
        if col is None:
            break  # optimal
        # Minimum ratio test
        ratios = []
        for i in range(m - 1):
            if tableau[i, col] > 1e-8:
                ratios.append((tableau[i, -1] / tableau[i, col], i))
        if not ratios:
            return "Solution non bornée."
        _, row = min(ratios)
        pivot(tableau, row, col)
        basis[row] = col

    solution = np.zeros(n - 1)
    for i, bi in enumerate(basis):
        solution[bi] = tableau[i, -1]
    obj = tableau[-1, -1]
    return f"Solution optimale : x = {solution.tolist()}\nValeur de la fonction objectif : {obj}"