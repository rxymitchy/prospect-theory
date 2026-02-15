import numpy as np
import copy
import itertools


def compute_ptne_equilibrium(U, pt, p1_type, p2_type):
    """Compute the cpt equilibrium using the gauss seidel fixed point iteration.
       NOTE FOR CODE REVIEW: This function is only returning pure equilibria, and 
       even a pure equilibrium for crawford's game. This was unexpected, so please note that
       current surprise.
    """

    # Define equilibria variables
    mixed_equil = dict()

    # Define Util Funtion:
    def util_func(values, probs, player_type):
        if player_type == "EU":
            return probs @ values

        elif player_type == "PT":
            return pt.expected_pt_value(values, probs)

    # Define p, q from z, where p, q are probs that player 1/2 play action 0 respectively 
    starting_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    init_list = itertools.product(starting_points, starting_points)

    # Define max p1 and p2 vals
    max_p1, max_p2 = 0, 0

    max_steps = 300

    # Define eps for tie breaks
    eps = 1e-8

    # Alpha for damping
    alpha = 0.25

    # Set the discretized iteration
    iter_list = np.linspace(0, 1, 200)

    # Set the flattened payoffs for players:
    p_1_payoffs = [U[0, 0, 0], U[1, 0, 0], U[0, 1, 0], U[1, 1, 0]]
    p_2_payoffs = [U[0, 0, 1], U[1, 0, 1], U[0, 1, 1], U[1, 1, 1]]

    for init_p, init_q in init_list:
        print(f'init_p: {init_p}, init_q: {init_q} search start')
        p, q = init_p, init_q
        current_step = 0

        while current_step < max_steps:
            current_step += 1

            # For Player 1, iterate over p and keep q fixed
            max_value_1 = -np.inf
            max_pval = p

            for p_i in iter_list:
                # Forming a full prospect across all 4 outcomes
                probs = [p_i * q, (1 - p_i) * q, p_i * (1 - q), (1 - p_i) * (1 - q)]
                value = util_func(p_1_payoffs, probs, p1_type)
                if value > max_value_1 + eps:
                    max_value_1 = value
                    max_pval = p_i
     
                # Tie breaking: choose p val closest to current p (minimize size of step)
                elif abs(value - max_value_1) <= eps:
                    # Choose p val closest to the current p
                    p_old_dist, p_new_dist = abs(max_pval - p), abs(p_i - p) 
                    if p_old_dist > p_new_dist:
                        max_pval = p_i
                        max_value_1 = value

           
            # Guass Seidel
            p_old = p
            p = (1-alpha) * p + alpha * max_pval


            # Same for player 2, but keep p fixed and iterate over q:
            max_value_2 = -np.inf
            max_qval = q

            for q_i in iter_list:
                probs = [p * q_i, (1 - p) * q_i, p * (1 - q_i), (1 - p) * (1 - q_i)]
                value = util_func(p_2_payoffs, probs, p2_type)
                if value > max_value_2 + eps:
                    max_value_2 = value
                    max_qval = q_i
            
                # Tie breaking: choose q val closest to current q (minimize size of step)
                elif abs(value - max_value_2) <= eps:
                    # Choose q val closest to the current q
                    q_old_dist, q_new_dist = abs(max_qval - q), abs(q_i - q)
                    if q_old_dist > q_new_dist:
                        max_qval = q_i
                        max_value_2 = value 

            # Gause Seidel
            q_old = q
            q = (1-alpha) * q + alpha * max_qval

            # Equilibrium check and logging
            tol = 1e-8
            if abs(p_old - p) < tol and abs(q_old - q) < tol:
                key = f'p={round(p, 6)}, q={round(q, 6)}'
                if key not in mixed_equil.keys():
                    mixed_equil[key] = []
                mixed_equil[key].append((init_p, init_q)) 
                break

    
    return mixed_equil
