# ============================================================================
# 3. EQUILIBRIUM FINDING FUNCTIONS
# ============================================================================

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

