# ============================================================================
# 4. IMPLEMENTATION VERIFICATION FUNCTIONS
# ============================================================================

def verify_pt_implementation():
    """Verify PT implementation matches theoretical properties"""

    checks = []

    # Create PT object with standard parameters
    pt = ProspectTheory(lambd=2.25, alpha=0.88, gamma=0.61, r=0)

    # 1. Check probability weighting shape (inverse S)
    p_low = 0.1
    p_high = 0.9
    w_low = pt.probability_weighting(p_low)
    w_high = pt.probability_weighting(p_high)

    checks.append({
        'test': 'Probability Weighting (low p)',
        'result': f'w({p_low}) = {w_low:.3f} > {p_low}',
        'passed': w_low > p_low,
        'required': f'w(p) > p for small p (inverse S-shape)'
    })

    checks.append({
        'test': 'Probability Weighting (high p)',
        'result': f'w({p_high}) = {w_high:.3f} < {p_high}',
        'passed': w_high < p_high,
        'required': f'w(p) < p for large p (inverse S-shape)'
    })

    # 2. Check loss aversion
    gain = 100
    loss = -100
    value_gain = pt.value_function(gain)
    value_loss = pt.value_function(loss)
    loss_aversion_ratio = abs(value_loss / value_gain)

    checks.append({
        'test': 'Loss Aversion',
        'result': f'λ_effective = {loss_aversion_ratio:.2f} (λ={pt.lambd})',
        'passed': loss_aversion_ratio > 1.5,  # Should be > 1 due to loss aversion
        'required': 'Losses weighted more heavily than gains'
    })

    # 3. Check diminishing sensitivity
    gain_small = 10
    gain_large = 100
    value_gain_small = pt.value_function(gain_small)
    value_gain_large = pt.value_function(gain_large)

    # Marginal value should decrease
    marginal_small = value_gain_small / gain_small
    marginal_large = (value_gain_large - value_gain_small) / (gain_large - gain_small)

    checks.append({
        'test': 'Diminishing Sensitivity (gains)',
        'result': f'Marginal value: {marginal_small:.3f} > {marginal_large:.3f}',
        'passed': marginal_small > marginal_large,
        'required': 'Decreasing marginal utility for gains'
    })

    # 4. Check reference point
    pt_with_ref = ProspectTheory(lambd=2.25, alpha=0.88, gamma=0.61, r=50)
    payoff = 60  # 10 units above reference
    value = pt_with_ref.value_function(payoff - pt_with_ref.r)

    checks.append({
        'test': 'Reference Point',
        'result': f'Payoff {payoff}, ref {pt_with_ref.r}, value = {value:.2f}',
        'passed': value > 0,  # Should be positive (gain)
        'required': 'Values calculated relative to reference point'
    })

    return checks, pt

def fine_search_ochs_game():
    """Ultra-fine grid search for PT-EB in Ochs game"""

    # Ochs game as defined
    ochs_game = np.array([
        [[0, 0], [5, 8]],
        [[8, 5], [0, 0]]
    ])

    # Parameters from thesis claim for Ochs game
    pt_params = [
        {'lambd': 0, 'alpha': 1.0, 'gamma': 1.0, 'r': 1.0, 'name': 'Thesis claim'},
        {'lambd': 0, 'alpha': 1.0, 'gamma': 0.61, 'r': 1.0, 'name': 'With prob weighting'},
        {'lambd': 1.0, 'alpha': 1.0, 'gamma': 1.0, 'r': 1.0, 'name': 'With loss aversion'},
    ]

    results = []

    for params in pt_params:
        pt = ProspectTheory(**{k: v for k, v in params.items() if k != 'name'})

        # Ultra-fine grid around thesis claim: (0.5, 0.05)
        p_range = np.linspace(0.4, 0.6, 201)  # 0.001 steps
        q_range = np.linspace(0.0, 0.15, 151) # 0.001 steps

        best_eq = None
        min_deviation = float('inf')

        for p in p_range:
            for q in q_range:
                # Calculate PT best responses
                br1 = find_pt_best_response(q, ochs_game, pt, player=0)
                br2 = find_pt_best_response(p, ochs_game, pt, player=1)

                # Check if (p, br1) and (br2, q) are close
                deviation = abs(p - br1) + abs(q - br2)

                if deviation < min_deviation:
                    min_deviation = deviation
                    best_eq = (p, q, br1, br2, deviation)

        if best_eq:
            p, q, br1, br2, dev = best_eq
            results.append({
                'params': params['name'],
                'p': p,
                'q': q,
                'br1': br1,
                'br2': br2,
                'deviation': dev,
                'close_to_claim': abs(p-0.5) < 0.05 and abs(q-0.05) < 0.05
            })

    return results
