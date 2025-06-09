import numpy as np

class SimplexeSolver:
    def __init__(self):
        self.MAX_ITER = 100
        
    def solve(self, constraints, objective, opt_type="max"):
        # Convert to standard form
        num_vars = len(objective)
        num_constraints = len(constraints)
        
        # Create tableau
        tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
        
        # Fill constraints
        for i in range(num_constraints):
            tableau[i, :num_vars] = constraints[i][:-1]
            tableau[i, num_vars + i] = 1  # Slack variables
            tableau[i, -1] = constraints[i][-1]  # RHS
        
        # Fill objective
        if opt_type == "max":
            tableau[-1, :num_vars] = [-x for x in objective]
        else:
            tableau[-1, :num_vars] = objective
        
        # Solve
        iter_count = 0
        while iter_count < self.MAX_ITER:
            # Check if optimal
            if opt_type == "max":
                if all(x >= 0 for x in tableau[-1, :-1]):
                    break
            else:
                if all(x <= 0 for x in tableau[-1, :-1]):
                    break
            
            # Select pivot column
            if opt_type == "max":
                pivot_col = np.argmin(tableau[-1, :-1])
            else:
                pivot_col = np.argmax(tableau[-1, :-1])
            
            # Select pivot row
            ratios = []
            for i in range(num_constraints):
                if tableau[i, pivot_col] > 0:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(np.inf)
            
            if all(r == np.inf for r in ratios):
                return "Problem is unbounded"
            
            pivot_row = np.argmin(ratios)
            
            # Pivot
            pivot_val = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_val
            
            for i in range(num_constraints + 1):
                if i != pivot_row:
                    tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
            
            iter_count += 1
        
        # Extract solution
        solution = {}
        for j in range(num_vars):
            col = tableau[:, j]
            if sum(col == 1) == 1 and sum(col == 0) == num_constraints:
                row = np.where(col == 1)[0][0]
                solution[f"x{j+1}"] = tableau[row, -1]
            else:
                solution[f"x{j+1}"] = 0
        
        solution["optimal_value"] = tableau[-1, -1] if opt_type == "max" else -tableau[-1, -1]
        
        return solution