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

def find_pt_nash_equilibrium(payoff_matrix, pt, grid_points=21):
    """
    Find Prospect Theory Nash Equilibrium using grid search

    Parameters:
    -----------
    payoff_matrix : ndarray, game payoff matrix
    pt : ProspectTheory object
    grid_points : int, number of grid points for search

    Returns:
    --------
    tuple or None: (p, q) if PT-NE found, None otherwise
    """
    best_eq = None
    min_deviation = float('inf')

    # Create grid
    p_values = np.linspace(0, 1, grid_points)
    q_values = np.linspace(0, 1, grid_points)

    for p in p_values:
        for q in q_values:
            # Calculate best responses
            br1 = find_pt_best_response(q, payoff_matrix, pt, player=0)
            br2 = find_pt_best_response(p, payoff_matrix, pt, player=1)

            # Check deviation from equilibrium
            deviation = abs(p - br1) + abs(q - br2)

            if deviation < min_deviation:
                min_deviation = deviation
                best_eq = (p, q, br1, br2, deviation)

    # If deviation is small enough, return as equilibrium
    if min_deviation < 0.05:  # 5% tolerance
        p, q, _, _, _ = best_eq
        return (p, q)
    else:
        return None

def find_pt_equilibrium_in_beliefs(payoff_matrix, pt, grid_points=51):
    """
    Find Prospect Theory Equilibrium-in-Beliefs

    Parameters:
    -----------
    payoff_matrix : ndarray, game payoff matrix
    pt : ProspectTheory object
    grid_points : int, number of grid points for search

    Returns:
    --------
    tuple or None: (p, q) if PT-EB found, None otherwise
    """
    # PT-EB: Each player optimizes given their beliefs about other's strategy
    # We look for fixed point in belief space

    best_eq = None
    min_deviation = float('inf')

    # Create belief grid
    p_beliefs = np.linspace(0, 1, grid_points)
    q_beliefs = np.linspace(0, 1, grid_points)

    for p_belief in p_beliefs:  # Player 2's belief about player 1
        for q_belief in q_beliefs:  # Player 1's belief about player 2
            # Given beliefs, players choose optimal strategies
            p_strat = find_pt_best_response(q_belief, payoff_matrix, pt, player=0)
            q_strat = find_pt_best_response(p_belief, payoff_matrix, pt, player=1)

            # Check if beliefs match chosen strategies
            deviation = abs(p_belief - p_strat) + abs(q_belief - q_strat)

            if deviation < min_deviation:
                min_deviation = deviation
                best_eq = (p_strat, q_strat, p_belief, q_belief, deviation)

    # Return if beliefs are consistent with strategies
    if min_deviation < 0.05:  # 5% tolerance
        p_strat, q_strat, _, _, _ = best_eq
        return (p_strat, q_strat)
    else:
        return None

def find_pt_best_response(opponent_strategy, payoff_matrix, pt, player=0):
    """
    Find best response given opponent's strategy

    Parameters:
    -----------
    opponent_strategy : float, probability opponent plays strategy 0
    payoff_matrix : ndarray, game payoff matrix
    pt : ProspectTheory object
    player : int, 0 for row player, 1 for column player

    Returns:
    --------
    float: optimal probability to play strategy 0
    """
    best_payoff = -float('inf')
    best_response = 0.5  # Default

    # Test discrete strategies first (simplified)
    for p_test in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if player == 0:
            payoff = pt.calculate_pt_payoff(p_test, opponent_strategy, payoff_matrix, 0)
        else:
            payoff = pt.calculate_pt_payoff(opponent_strategy, p_test, payoff_matrix, 1)

        if payoff > best_payoff:
            best_payoff = payoff
            best_response = p_test

    return best_response
