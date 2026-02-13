import numpy as np
import random


def compute_ptne_equilibrium(U, pt, p1_type, p2_type):
    """Compute the cpt equilibrium using the semi smooth newton method.
       First, check for pure equilibria, then check for mixed strategies with newton"""
    # Define equilibria variables
    mixed_equil = []

    # Define Util Funtion:
    def util_func(values, probs, player_type):
        if player_type == "EU":
            return probs @ values

        elif player_type == "PT":
            return pt.expected_pt_value(values, probs)

    # Define p, q from z
    p, q = 0.5, 0.5

    # Define max p1 and p2 vals
    max_p1, max_p2 = 0, 0

    max_steps = 1000
    current_step = 0

    # Define eps for tie breaks
    eps = 1e-8

    # Set the discretized iteration
    iter_list = np.linspace(0, 1, 250)

    # Set the flattened payoffs for players:
    p_1_payoffs = [U[0, 0, 0], U[1, 0, 0], U[0, 1, 0], U[1, 1, 0]]
    p_2_payoffs = [U[0, 0, 1], U[1, 0, 1], U[0, 1, 1], U[1, 1, 1]]

    while current_step < max_steps:
        current_step += 1

        # For Player 1, iterate over p and keep q fixed
        max_value_1 = -np.inf
        max_pval = 0.01
        max_pvals = []

        for p_i in iter_list:
	    # Forming a full prospect across all 4 outcomes
            probs = [p_i * q, (1 - p_i) * q, p_i * (1 - q), (1 - p_i) * (1 - q)]
            value = util_func(p_1_payoffs, probs, p1_type)
            if value > max_value_1 + eps:
                max_value_1 = value
                max_pval = p_i
 
                # Reset the list, there's a new max in town
                max_pvals = [p_i]

            elif abs(value - max_value_1) <= eps:
                if random.random() < 0.5:
                    max_value_1 = value
                    max_pval = p_i

                # Append the tie
                max_pvals.append(p_i)
       
        # Guass Seidel
        p_old = p
        p = max_pval


        # Same for player 2, but keep p fixed and iterate over q:
        max_value_2 = -np.inf
        max_qval = 0.01
        max_qvals = []

        for q_i in iter_list:
            probs = [p * q_i, (1 - p) * q_i, p * (1 - q_i), (1 - p) * (1 - q_i)]
            value = util_func(p_2_payoffs, probs, p2_type)
            if value > max_value_2 + eps:
                max_value_2 = value
                max_qval = q_i
	    
                # Reset max pvals list
                max_qvals = [q_i]

            elif abs(value - max_value_2) <= eps:
                if random.random() < 0.5:
                    max_value_2 = value
                    max_qval = q_i

                # Add the new tie
                max_qvals.append(q_i)

        # Gause Seidel
        q_old = q
        q = max_qval

        if p_old == p and q_old == q:
            mixed_equil.append((p, q)) 
            break

    
    return mixed_equil
