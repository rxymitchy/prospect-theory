from .ProspectTheory import ProspectTheory
import numpy as np
import itertools

def compute_eb_equilibrium(U, pt, p1_type, p2_type):
    """Compute the cpt equilibrium using the semi smooth newton method.
       First, check for pure equilibria, then check for mixed strategies with newton""" 
    # Define equilibria variables
    pure_equil, mixed_equil = [], dict()

    # Define Util Funtion:
    def util_func(values, probs, player_type):
        if player_type == "EU":
            return probs @ values

        elif player_type == "PT":
            return pt.expected_pt_value(values, probs)

    # First, check for pure:
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p_1_a_1, p_1_a_2 = U[i, j, 0], U[1-i, j, 0]
            if p1_type == "PT":
                p_1_a_1, p_1_a_2 = pt.value_function(p_1_a_1), pt.value_function(p_1_a_2)    

            p_2_a_1, p_2_a_2 = U[i, j, 1], U[i, 1-j, 1]
            if p2_type == "PT":
                p_2_a_1, p_2_a_2 = pt.value_function(p_2_a_1), pt.value_function(p_2_a_2)
    
            if p_1_a_1 >= p_1_a_2 and p_2_a_1 >= p_2_a_2:
                pure_equil.append((i, j))

    # Now, mixed strategies with newtown semismooth:
    # Start with many seeds to explore the space (why not?)
    starting_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    init_list = itertools.product(starting_points, starting_points) 

    # Iterate through seed pairs, append to mixed equil
    for p, q in init_list:
        # Stopping condition
        max_tries = 1000
        counter = 0
        z = np.array([p, q])
        while counter < max_tries:
            counter += 1

            try:
                z, is_root = semismooth_newton(U, z, util_func, p1_type, p2_type)

            except ValueError as e:
                print(f'{e}')
                break

            if is_root:
                z_round = np.round(z, 5)
                key = f'equil = {tuple(z_round)}'
                if key not in mixed_equil.keys():
                    mixed_equil[key] = [f'p={p}, q={q}']
                else:
                    mixed_equil[key].append(f'p={p}, q={q}')
                break

    return pure_equil, mixed_equil



def semismooth_newton(U, z, util_func, p1_type, p2_type, eps=1e-6):
    # A semismooth newton solver for pt equilibrium
    # Step 1) Define starting conditions for each strategy (e.g. (0.5, 0.5))
    p, q = z
    # Step 2) Map the probabilities z = p, q into F space
    # F is the difference vector between actions for each player
    # That is, CPT( Value_A | prob) - CPT (Value_B | prob)

    # Calculate F_i for each player via the indifference equation u(A_1) - u(A_2)
    # Player 1 is rows, player 2 is columns
    row_a_1, row_a_2 = np.array([U[0, 0, 0], U[0, 1, 0]]), np.array([U[1, 0, 0], U[1, 1, 0]])
    col_a_1, col_a_2 = np.array([U[0, 0, 1], U[1, 0, 1]]), np.array([U[0, 1, 1], U[1, 1, 1]])
    
    # Define the prob vectors for both: p1 is the probs player 2 players action 0 or 1, and vice versa
    p_1_probs, p_2_probs = np.array([q, 1-q]), np.array([p, 1-p])
     
    player_1_F = util_func(row_a_1, p_1_probs, p1_type) - util_func(row_a_2, p_1_probs, p1_type)
    player_2_F = util_func(col_a_1, p_2_probs, p2_type) - util_func(col_a_2, p_2_probs, p2_type) 
    F_z = np.array([player_1_F, player_2_F])

    if abs(F_z[0]) < eps and abs(F_z[1]) < eps:
        return z, True

    # Step 3) Compute the Jacobian J for the mapping p,q -> F
    jacobian = compute_jacobian(p, q, util_func, U, p1_type, p2_type) 

    # Step 4) Solve for delta s.t. Jdelta = -F(z)
    delta = np.linalg.solve(jacobian, -F_z)
   
    # Step 5) clip and return the updated p, q values
    z += delta 
    z = np.clip(z, 0, 1)

    return z, False 

def compute_jacobian(p, q, util_func, U, p1_type, p2_type, eps=1e-6):
    ''' A numerical computation of the jacobian matrix for the semismooth newton method
    '''
    # Check if we are near boundary
    if p + eps > 1 or p - eps < 0 or q + eps > 1 or q - eps < 0:
        raise ValueError(f"probability exceeded threshold: p = {p}, q = {q}, eps = {eps}")

    # Initialize Jacobian Matrix
    J = np.zeros((2, 2))

    # Redefine actions:
    row_a_1, row_a_2 = np.array([U[0, 0, 0], U[0, 1, 0]]), np.array([U[1, 0, 0], U[1, 1, 0]])
    col_a_1, col_a_2 = np.array([U[0, 0, 1], U[1, 0, 1]]), np.array([U[0, 1, 1], U[1, 1, 1]])

    # Compute F1 (only with q, F1 does not depend on p)
    p_1_probs_plus, p_1_probs_minus = np.array([q + eps, 1 - (q + eps)]), np.array([q - eps, 1 - (q - eps)])

    F1_plus = util_func(row_a_1, p_1_probs_plus, p1_type) - util_func(row_a_2, p_1_probs_plus, p1_type)
    F1_minus = util_func(row_a_1, p_1_probs_minus, p1_type) - util_func(row_a_2, p_1_probs_minus, p1_type)
    F1_delta = (F1_plus - F1_minus) / (2 * eps)

    # Compute F2 (only with p, F2 does not depend on F1)
    p_2_probs_plus, p_2_probs_minus = np.array([p + eps, 1 - (p + eps)]), np.array([p - eps, 1 - (p - eps)])

    F2_plus = util_func(col_a_1, p_2_probs_plus, p2_type) - util_func(col_a_2, p_2_probs_plus, p2_type)
    F2_minus = util_func(col_a_1, p_2_probs_minus, p2_type) - util_func(col_a_2, p_2_probs_minus, p2_type)
    F2_delta = (F2_plus - F2_minus) / (2 * eps)

    # Add to Jacobian
    J[0, 1], J[1, 0] = F1_delta, F2_delta

    return J
