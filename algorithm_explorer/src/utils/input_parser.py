def parse_adjacency_matrix(input_text):
    try:
        matrix = [list(map(float, line.split())) for line in input_text.strip().split('\n')]
        return matrix
    except ValueError:
        raise ValueError("Invalid input for adjacency matrix. Please ensure it is a valid matrix format.")

def parse_constraints(input_text):
    try:
        lines = input_text.strip().split('\n')
        constraints = [list(map(float, line.split())) for line in lines[:-1]]
        objective_function = list(map(float, lines[-1].split()))
        return constraints, objective_function
    except ValueError:
        raise ValueError("Invalid input for constraints. Please ensure it is a valid format.")

def parse_ford_fulkerson_input(input_text):
    try:
        lines = input_text.strip().split('\n')
        matrix = [list(map(float, line.split())) for line in lines[:-2]]
        source = int(lines[-2].strip())
        sink = int(lines[-1].strip())
        return matrix, source, sink
    except (ValueError, IndexError):
        raise ValueError("Invalid input for Ford-Fulkerson. Please ensure the format is correct.")

def parse_dijkstra_input(input_text):
    try:
        lines = input_text.strip().split('\n')
        matrix = [list(map(float, line.split())) for line in lines[:-1]]
        start_node = int(lines[-1].strip())
        return matrix, start_node
    except (ValueError, IndexError):
        raise ValueError("Invalid input for Dijkstra's Algorithm. Please ensure the format is correct.")