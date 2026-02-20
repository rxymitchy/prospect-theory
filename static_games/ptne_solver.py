from .ProspectTheory import ProspectTheory
import numpy as np
import itertools

def compute_ptne_equilibrium(U, pt, p1_type, p2_type):
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

    # First, check for pure. For CPT NE, it becomes degenerate because the opp beliefs are 0, 1. 
    # We calculate the prospect using the certainty given to us by the loops 

    # PT NE is non linear in our strategy space, so we cant make the assumption that 0 >= 1 or vice versa
    # We need to account for the possibility that a midpoint between them is actually the best response, because our strategy space is nonlinear
    grid_search = np.linspace(0, 1, 201)

    def V1(q, p1_type):
        '''Finds the max point in the curve given opponent probs'''
        max_v = -np.inf
        # Our values stay static
        values = np.array([U[0, 0, 0], U[1, 0, 0], U[0, 1, 0], U[1, 1, 0]])

        # But we need to compute new probs for each of our possible p values
        # then get our max value (we don't care about tracking p, just the max value)
        for p in grid_search:
            probs = np.array([p * q, (1 - p) * q, p * (1 - q), (1 - p) * (1 - q)])
            V = util_func(values, probs, p1_type)
            if V > max_v:
                max_v = V 
        return max_v

    def V2(p, p2_type):
        '''Finds the max point in the curve given opponent probs'''
        max_v = -np.inf
        # Our values stay static
        values = np.array([U[0, 0, 1], U[1, 0, 1], U[0, 1, 1], U[1, 1, 1]])

        # But we need to compute new probs for each of our possible p values
        # then get our max value (we don't care about tracking p, just the max value)
        for q in grid_search:
            probs = np.array([p * q, (1 - p) * q, p * (1 - q), (1 - p) * (1 - q)])
            V = util_func(values, probs, p2_type)
            if V > max_v:
                max_v = V
        return max_v

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p = 1.0 if i == 0 else 0.0 # Prob P1 plays action 0
            q = 1.0 if j == 0 else 0.0 # Prob P2 plays action 0

            p1_probs = np.array([p * q, (1 - p) * q, p * (1 - q), (1 - p) * (1 - q)])
            p2_probs = p1_probs

            # We check what the values of staying are across opponent actions, 
            # And then we get the value if we deviate for each action

            p1_stay = np.array([U[0, 0, 0], U[1, 0, 0], U[0, 1, 0], U[1, 1, 0]]) 

            p2_stay = np.array([U[0, 0, 1], U[1, 0, 1], U[0, 1, 1], U[1, 1, 1]])

            # Then we can compute two lotteries with opp probs fixed and stay/dev measured
            v1_1, v1_2 = util_func(p1_stay, p1_probs, p1_type), V1(q, p1_type)
            v2_1, v2_2 = util_func(p2_stay, p2_probs, p2_type), V2(p, p2_type)

            tol = 1e-8

            # So if both players want to stay, we are at equilibrium
            if v1_1 >= v1_2 - tol and v2_1 >= v2_2 - tol:
                pure_equil.append((i, j))

    # Turn indices into probabilities
    for idx, (i, j) in enumerate(pure_equil):
        p = 1.0 if i == 0 else 0.0 # Prob P1 plays action 0
        q = 1.0 if j == 0 else 0.0 # Prob P2 plays action 0
        pure_equil[idx] = (p, q)


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
                print(f'Semismooth failed, error: {e}')
                break

            if is_root:
                # Validate whether root is equilibrium, same as above
                p_star, q_star = z
  
                # Ignore pure equilibria (computed above)
                if (abs(p_star - 1.0) <= 1e-5 or p_star <= 1e-5) and (abs(q_star - 1.0) <= 1e-5 or q_star <= 1e-5):
                    break

                # Compute the stay (proposed roots) and deviate (any other p/q value) values

                probs = np.array([p_star * q_star, (1 - p_star) * q_star, p_star * (1 - q_star), (1 - p_star) * (1 - q_star)])

                p1_stay = np.array([U[0, 0, 0], U[1, 0, 0], U[0, 1, 0], U[1, 1, 0]]) # row payoffs
                p2_stay = np.array([U[0, 0, 1], U[1, 0, 1], U[0, 1, 1], U[1, 1, 1]]) # col payoffs

                '''
                Then we can compute two lotteries with opp probs fixed and stay/dev measured
                 vi_1 is just the value according to the player's preference of being at the proposed prob
                 vi_2 uses the same grid searh functions for the pure equilibria search
                 and passes the opposing players prob (to stay fixed)
                 and then iterates over 201 of the player's action 0 probs. 
                 Conceptually, we are checking the proposed maximum as compared to all other points in the strategy space

                 I think about it as two see saws, in NE and EB we have flat see saws, but in PTNE the seesaws are curved
                 and the curve changes with the probability changes, so we need to do a thorough check to make sure
                 we are at a max because the entire landscape is constantly shifting. 
                '''
                v1_1, v1_2 = util_func(p1_stay, probs, p1_type), V1(q_star, p1_type)
                v2_1, v2_2 = util_func(p2_stay, probs, p2_type), V2(p_star, p2_type)

                tol = 1e-8

                # So if both players want to stay, we are at equilibrium
                if v1_1 >= v1_2 - tol and v2_1 >= v2_2 - tol:
                    # Track
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
    Here F(z) is the residual of the gradient ascent using the derivative of the value function provided.
    Formally: F = p - clip(p + tau * derivative), where p is our prob and tau is a step size paramete.
    We are looking for the point where the derivative is 0 (external scans prevent saddle points and minima)
    
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

    # Calculate F_i for each player using the residual from a gradient ascent step. 
    # E.g.: F_1 = p - clip(p + tau * derivative), where tau is a step size parameter 
    # Player 1 is rows, player 2 is columns
    p1_payoffs = np.array([U[0, 0, 0], U[1, 0, 0], U[0, 1, 0], U[1, 1, 0]])
    p2_payoffs = np.array([U[0, 0, 1], U[1, 0, 1], U[0, 1, 1], U[1, 1, 1]])
     
    F_p = F(p, q, util_func, U, p1_payoffs, p1_type, 0, opt_p = True)
    F_q = F(p, q, util_func, U, p2_payoffs, p2_type, 1, opt_p = False)

    F_z = np.array([F_p, F_q])

    if abs(F_z[0]) < eps and abs(F_z[1]) < eps:
        return z, True

    # Step 3) Compute the Jacobian J for the mapping p,q -> F
    jacobian = compute_jacobian(p, q, util_func, U, p1_type, p2_type) 

    # Step 4) Solve for delta s.t. Jdelta = -F(z)
    delta = np.linalg.solve(jacobian, -F_z)
   
    # Step 5) clip and return the updated p, q values (with damping)
    alpha = 0.5 # Step wise damping paremeter
    z += alpha * delta
    z = np.clip(z, 0, 1)

    return z, False 

def F(p, q, util_func, U, player_action, p_type, p_id, opt_p=True, eps=1e-6, tau=0.1):
    ''' A numerical computation of the mapping from p, q to the derivative for the curve for the semismooth newton method
        Importantly, either p or q can be perturbed, but we hold static that p refers to player 1 and q to player 2
        p_id is used to determine whether the player is the column or row player
        opt_p is a boolean telling us whether to perturb p or q
        Importantly, p_id and opt_p are independent so they must be tracked separately (different cells of jacobian)
        tau is a step size parameter
    '''
    if opt_p: 
        if p - eps < 0.0:
            # forward difference
            p0, p1 = p, p + eps
        elif p + eps > 1.0:
            # backward difference
            p0, p1 = p - eps, p
        else:
            # central difference
            p0, p1 = p - eps, p + eps

        probs_plus = np.array([p1 * q, (1 - p1) * q, p1 * (1 - q), (1 - p1) * (1 - q)])
        probs_minus = np.array([p0 * q, (1 - p0) * q, p0 * (1 - q), (1 - p0) * (1 - q)])
        denom = p1 - p0

    else:
        if q - eps < 0.0:
            # forward difference
            q0, q1 = q, q + eps
        elif q + eps > 1.0:
            # backward difference
            q0, q1 = q - eps, q
        else:
            # central difference
            q0, q1 = q - eps, q + eps

        probs_plus = np.array([q1 * p, (1 - p) * q1, (1-q1) * p, (1 - q1) * (1 - p)])
        probs_minus = np.array([q0 * p, (1 - p) * q0, p * (1 - q0), (1 - q0) * (1 - p)])
        denom = q1 - q0


    # Compute F (only varying q, F1 does not depend optimize over p)
    F_plus = util_func(player_action, probs_plus, p_type)
    F_minus = util_func(player_action, probs_minus, p_type)

    F_delta = (F_plus - F_minus) / denom

    # Now we calculate the proposed step, and project back into the simplex. 
    if p_id == 0: # Row Player
        F = p - min(1, max(0, p + tau * F_delta))  

    else: # Col player
        F = q - min(1, max(0, q + tau * F_delta))

    return F

def compute_jacobian(p, q, util_func, U, p1_type, p2_type, eps=1e-5):
    ''' A numerical computation of the jacobian matrix for the semismooth newton method
        Unlike EB, we ned to use the full lottery instead of preselecting pure actions,
        and we optimize over both p and q.
    '''
    # Check if we are near boundary
    if p - eps < 0.0:
        # forward difference
        p0, p1 = p, p + eps
    elif p + eps > 1.0:
        # backward difference
        p0, p1 = p - eps, p
    else:
        # central difference
        p0, p1 = p - eps, p + eps

    # Get p denominator
    p_denom = p1 - p0

    if q - eps < 0.0:
        # forward difference
        q0, q1 = q, q + eps
    elif q + eps > 1.0:
        # backward difference
        q0, q1 = q - eps, q
    else:
        # central difference
        q0, q1 = q - eps, q + eps

    # Set q denominator
    q_denom = q1 - q0

    # Initialize Jacobian Matrix
    J = np.zeros((2, 2))

    # Redefine actions:
    player_1_action = np.array([U[0, 0, 0], U[1, 0, 0], U[0, 1, 0], U[1, 1, 0]])
    player_2_action = np.array([U[0, 0, 1], U[1, 0, 1], U[0, 1, 1], U[1, 1, 1]])

    # Compute F1/p
    F1_p_plus = F(p1, q, util_func, U, player_1_action, p1_type, 0, opt_p = True)
    F1_p_minus = F(p0, q, util_func, U, player_1_action, p1_type, 0, opt_p = True)
    F1_p_delta = (F1_p_plus - F1_p_minus) / p_denom

    # Add F1/p to the Jacobian:
    J[0,0] = F1_p_delta

    # Compute F1/q
    F1_q_plus = F(p, q1, util_func, U, player_1_action, p1_type, 0, opt_p = False)
    F1_q_minus = F(p, q0, util_func, U, player_1_action, p1_type, 0, opt_p = False)
    F1_q_delta = (F1_q_plus - F1_q_minus) / q_denom

    # Add F1/q to the Jacobian:
    J[0, 1] = F1_q_delta

    # Compute F2/p
    F2_p_plus = F(p1, q, util_func, U, player_2_action, p2_type, 1, opt_p = True)
    F2_p_minus = F(p0, q, util_func, U, player_2_action, p2_type, 1, opt_p = True)
    F2_p_delta = (F2_p_plus - F2_p_minus) / p_denom

    # Add F2/p to the Jacobian:
    J[1, 0] = F2_p_delta

    # Compute F2/q
    F2_q_plus = F(p, q1, util_func, U, player_2_action, p2_type, 1, opt_p = False)
    F2_q_minus = F(p, q0, util_func, U, player_2_action, p2_type, 1, opt_p = False)
    F2_q_delta = (F2_q_plus - F2_q_minus) / q_denom

    # Add to Jacobian
    J[1, 1] = F2_q_delta

    return J
