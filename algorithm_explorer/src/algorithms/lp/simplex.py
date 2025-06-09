def simplex(c, A, b):
    from scipy.optimize import linprog

    # Solve the linear programming problem
    res = linprog(c, A_ub=A, b_ub=b, method='highs')

    if res.success:
        return res.x, res.fun
    else:
        raise ValueError("The linear programming problem could not be solved.")

# Example usage (uncomment to test):
# c = [-1, -2]  # Coefficients for the objective function
# A = [[2, 1], [1, 1], [1, 0]]  # Coefficients for the inequality constraints
# b = [20, 16, 10]  # Right-hand side of the inequality constraints
# solution, objective_value = simplex(c, A, b)
# print("Optimal solution:", solution)
# print("Optimal value:", objective_value)