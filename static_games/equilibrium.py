from .ProspectTheory import ProspectTheory
import numpy as np

def find_classical_ne(payoff_matrix):
    """Find classical Nash equilibria analytically for 2x2 games"""
    pure_NE = []
    mixed_NE = None

    # Check pure strategy NE
    for i in [0, 1]:  # Player 1 strategies
        for j in [0, 1]:  # Player 2 strategies
            # Check if i is best response to j for player 1
            payoff_i_j = payoff_matrix[i, j, 0]
            payoff_other_j = payoff_matrix[1-i, j, 0]

            # Check if j is best response to i for player 2
            payoff_i_j2 = payoff_matrix[i, j, 1]
            payoff_i_other = payoff_matrix[i, 1-j, 1]

            if payoff_i_j >= payoff_other_j and payoff_i_j2 >= payoff_i_other:
                pure_NE.append((i, j))

    # Find mixed strategy NE (if exists)
    # For 2x2 games: p such that player 2 is indifferent
    # P = B/A, which is how we solve the indifference equation
    '''
    Derivation (only player 2 for brevity):
    player 2 indifference = p(U_0,0) + (1-p)(U_1,0) = p(U_0,1) + (1-p)(U_1,1) # Rows change on each side, columns are constant on each side
    Simplied to: p(U_0,0) + U_1,0 - p(U_1,0) = p(U_0,1) + U_1,1 - p(U_1,1)
    Get P on left side: p(U_0,0 - U_1,0 - U_0,1 + U_1,1) # This is A = U_1,1 - U_1,0 # This is B
    '''
    A = payoff_matrix[0, 0, 1] - payoff_matrix[0, 1, 1] - payoff_matrix[1, 0, 1] + payoff_matrix[1, 1, 1]
    B = payoff_matrix[1, 1, 1] - payoff_matrix[1, 0, 1]

    if abs(A) > 1e-10:
        p_mixed = B / A
        # q such that player 1 is indifferent
        # Rewrote this to match the correct theory (D should be 1,1 - 0, 1)
        C = payoff_matrix[0, 0, 0] - payoff_matrix[1, 0, 0] - payoff_matrix[0, 1, 0] + payoff_matrix[1, 1, 0]
        D = payoff_matrix[1, 1, 0] - payoff_matrix[0, 1, 0]

        if abs(C) > 1e-10:
            q_mixed = D / C

            # Check if valid probabilities
            if 0 <= p_mixed <= 1 and 0 <= q_mixed <= 1:
                # Check if it's not already a pure strategy
                if not (abs(p_mixed) < 1e-10 or abs(p_mixed-1) < 1e-10 or
                        abs(q_mixed) < 1e-10 or abs(q_mixed-1) < 1e-10):
                    mixed_NE = (p_mixed, q_mixed)

    if not pure_NE and mixed_NE is None:
        raise RuntimeError("Equilibrium exists theoretically, but solver missed it.")


    return pure_NE, mixed_NE

def compute_cpt_equilibrium(payoff_matrix):
    """Compute the cpt equilibrium using the semi smooth newton method.
       First, check for pure equilibria, then check for mixed strategies with newton""" 
    # Define equilibria variables
    pure_equil, mixed_equil = None, None

    # First, check for pure:
    




def semismooth_newton(U, z, util_func, eps=1e-6):
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
     
    player_1_F = util_func(row_a_1, p_1_probs) - util_func(row_a_2, p_1_probs)
    player_2_F = util_func(col_a_1, p_2_probs) - util_func(col_a_2, p_2_probs) 
    F_z = np.array([player_1_F, player_2_F])

    if abs(F_z[0]) < eps and abs(F_z[1]) < eps:
        return z, True

    # Step 3) Compute the Jacobian J for the mapping p,q -> F
    jacobian = compute_jacobian(p, q, util_func, U) 

    # Step 4) Solve for delta s.t. Jdelta = -F(z)
    delta = np.linalg.solve(jacobian, -F_z)
   
    # Step 5) clip and return the updated p, q values
    z += delta 
    z = np.clip(z, 0, 1)

    return z, False 

def compute_jacobian(p, q, util_func, U, eps=1e-6):
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

    F1_plus = util_func(row_a_1, p_1_probs_plus) - util_func(row_a_2, p_1_probs_plus)
    F1_minus = util_func(row_a_1, p_1_probs_minus) - util_func(row_a_2, p_1_probs_minus)
    F1_delta = (F1_plus - F1_minus) / (2 * eps)

    # Compute F2 (only with p, F2 does not depend on F1)
    p_2_probs_plus, p_2_probs_minus = np.array([p + eps, 1 - (p + eps)]), np.array([p - eps, 1 - (p - eps)])
    F2_plus = util_func(col_a_1, p_2_probs_plus) - util_func(col_a_2, p_2_probs_plus)
    F2_minus = util_func(col_a_1, p_2_probs_minus) - util_func(col_a_2, p_2_probs_minus)
    F2_delta = (F2_plus - F2_minus) / (2 * eps)

    # Add to Jacobian
    J[0, 1], J[1, 0] = F1_delta, F2_delta

    return J
