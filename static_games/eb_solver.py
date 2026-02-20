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

    # First, check for pure. For CPT EB, it becomes degenerate because the opp beliefs are 0, 1. 
    # We calculate the prospect using the certainty given to us by the loops 

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p = 1.0 if i == 0 else 0.0 # Prob P1 plays action 0
            q = 1.0 if j == 0 else 0.0 # Prob P2 plays action 0

            probs1, probs2 = np.array([q, 1-q]), np.array([p, 1-p]) 

            # We check what the values of staying are across opponent actions, 
            # And then we get the value if we deviate for each action
            # So, to explain indexing, stay keeps i fixed (where we are in the loop for p1) and hardcodes both p2 actions wrt the probs
            # THen deviate hardcodes the current player deviating (1-i) or (1-j), and then we operate over the opp actions

            p1_stay = np.array([U[i, 0, 0], U[i, 1, 0]]) # i stays fixed, opp actions change
            p1_dev = np.array([U[1-i, 0, 0], U[1-i, 1, 0]]) # i gets inverted, opp actions change

            p2_stay = np.array([U[0, j, 1], U[1, j, 1]]) # j stays fixed, row changes
            p2_dev = np.array([U[0, 1-j, 1], U[1, 1-j, 1]]) # j gets inverted, row changes

            # Then we can compute two lotteries with opp probs fixed and stay/dev measured
            v1_1, v1_2 = util_func(p1_stay, probs1, p1_type), util_func(p1_dev, probs1, p1_type)
            v2_1, v2_2 = util_func(p2_stay, probs2, p2_type), util_func(p2_dev, probs2, p2_type)
    
            # So if both players want to stay, we are at equilibrium
            if v1_1 >= v1_2 and v2_1 >= v2_2:
                pure_equil.append((i, j))

    for idx, (i, j) in enumerate(pure_equil):
        p = 1.0 if i == 0 else 0.0 # Prob P1 plays action 0
        q = 1.0 if j == 0 else 0.0 # Prob P2 plays action 0
        pure_equil[idx] = (p, q) 

    # Now, mixed strategies with newtown semismooth:
    # Start with many seeds to explore the space (why not?)
    # The point is to make sure we aren't convrging to specific basins based on initialization
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
                print(f'Semismooth failed, error: {e}')
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
    '''
    A semismooth newton solver for cpt equilibrium of beliefs. We are looking for zeros in the F(z) function
    Here F(z) is the indifference (optimality?) condition val(action 1) = val (action 2), and we are looking for 
    the point where both player 1 and 2 are indifferent
    
    The basic formula is:
    1) initialize z = p, q
    2) get initial F(z) (if F(z) = 0, return
    3) Comute Jacobian (derivative matrix to see how small perturbations in p, q influence F)
        a) used finite difference, not anything analytical
    4) solve linear equation x = J^-1 (-F(z)) to find where the function hits 0
    5) update z with x, repeat if needed (outer loop)
    '''
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
    z = np.clip(z, 0, 1) # clip to avoid returning a value outside of the prob range

    return z, False 

def compute_jacobian(p, q, util_func, U, p1_type, p2_type, eps=1e-6):
    ''' 
    A numerical computation of the jacobian matrix for the semismooth newton method
    Uses finite difference with the indifference equation
    Do i need to add logic for forward and backward difference near boundaries? 
    I fugured that if we're near a boundary in the first place its a pure strategy response, 
    and the value error already catches it, so i think the only risk is pure/mixed equilibria being missed
    which seems unlikely anyways. 
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

    # Add to Jacobian, 0, 0 and 1, 1 are dF1/dp and dF2/dq, and since F1 only depends on q and F2 only 
    # depends on p, those derivatives will always be zero. 
    J[0, 1], J[1, 0] = F1_delta, F2_delta

    return J
