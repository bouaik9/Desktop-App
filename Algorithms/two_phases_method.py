import numpy as np
from scipy.optimize import linprog

class TwoPhaseSimplex:
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def solve(self, matrix):
        """
        Solves LP problem using two-phase simplex method.
        
        Args:
            matrix: numpy array where last row is objective function,
                   last column is RHS, and other elements are constraints
        
        Returns:
            dict: Contains solution status, optimal values, and objective value
        """
        try:
            # Convert and validate input
            matrix = np.array(matrix, dtype=float)
            if matrix.ndim != 2:
                raise ValueError("Input must be a 2D matrix")

            # Extract components
            self.c = -matrix[-1, :-1]  # Negative for maximization
            self.A = matrix[:-1, :-1]
            self.b = matrix[:-1, -1]
            
            # Check if two-phase method is needed
            if np.all(self.b >= 0):
                return self._solve_direct()
                
            return self._solve_two_phase()

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _solve_direct(self):
        """Solves LP directly when all RHS values are non-negative."""
        result = linprog(
            c=self.c,
            A_ub=self.A,
            b_ub=self.b,
            method='simplex'
        )
        
        return self._format_result(result)

    def _solve_two_phase(self):
        """Implements two-phase simplex method."""
        # Phase 1: Find initial feasible solution
        m, n = self.A.shape
        
        # Create artificial variables
        art_vars = np.eye(m)
        A_phase1 = np.hstack((self.A, art_vars))
        c_phase1 = np.hstack((np.zeros(n), np.ones(m)))
        
        # Solve Phase 1
        res_phase1 = linprog(
            c=c_phase1,
            A_eq=A_phase1,
            b_eq=self.b,
            method='simplex'
        )
        
        if not res_phase1.success or abs(res_phase1.fun) > self.epsilon:
            return {
                "success": False,
                "error": "No feasible solution exists"
            }
            
        # Phase 2: Solve original problem
        basic_vars = res_phase1.x[:n]
        
        res_phase2 = linprog(
            c=self.c,
            A_ub=self.A,
            b_ub=self.b,
            x0=basic_vars,
            method='simplex'
        )
        
        return self._format_result(res_phase2)

    def _format_result(self, result):
        """Formats the optimization result."""
        if not result.success:
            return {
                "success": False,
                "error": result.message
            }

        # Format solution values
        solution = [self._format_number(x) for x in result.x]
        obj_value = -self._format_number(result.fun)  # Negative because we maximized

        return f"La soltion optimale est :\n{solution},\n la valeur de la fonction objectif : {obj_value}\n nombre d'itérations : {result.nit}"
    

    def _format_number(self, value):
        """Formats numeric values with appropriate precision."""
        if abs(value) < self.epsilon:
            return 0
        if abs(value - round(value)) < self.epsilon:
            return int(round(value))
        return round(value, 4)

def two_phase_method(matrix):

    solver = TwoPhaseSimplex()
    return solver.solve(matrix)

