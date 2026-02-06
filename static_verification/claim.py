# ============================================================================
# 6. THESIS CLAIM VERIFICATION
# ============================================================================

def verify_thesis_claims(analysis_results):
    """Verify thesis claims based on analysis results"""

    claims = {
        'pt_ne_pathology': {
            'description': 'PT-NE may fail to exist (pathology)',
            'games': [],
            'supported': False
        },
        'pt_eb_when_ne_fails': {
            'description': 'PT-EB exists when PT-NE fails',
            'games': [],
            'supported': False
        },
        'pure_ne_identical': {
            'description': 'Pure strategy NE are identical under PT',
            'games': [],
            'supported': False
        },
        'mixed_ne_different': {
            'description': 'PT changes mixed strategy predictions',
            'games': [],
            'supported': False
        }
    }

    # Analyze each result
    for result in analysis_results:
        game_name = result['game']

        # Claim 1: PT-NE pathology
        if result['pt_ne'] is None:
            claims['pt_ne_pathology']['games'].append(game_name)
            claims['pt_ne_pathology']['supported'] = True

        # Claim 2: PT-EB when PT-NE fails
        if result['pt_ne'] is None and result['pt_eb'] is not None:
            claims['pt_eb_when_ne_fails']['games'].append(game_name)
            claims['pt_eb_when_ne_fails']['supported'] = True

        # Claim 3: Pure NE identical
        # Check if classical pure NE exist and PT preserves them
        if result['classical_pure']:
            # For each pure NE, check if it's stable under PT
            # (Simplified: assume pure strategies are preserved if PT-NE includes them)
            if result['pt_ne']:
                p_pt, q_pt = result['pt_ne']
                # Check if PT-NE is close to any pure strategy
                for pure_ne in result['classical_pure']:
                    p_pure, q_pure = pure_ne
                    if (abs(p_pt - p_pure) < 0.1 and abs(q_pt - q_pure) < 0.1):
                        claims['pure_ne_identical']['games'].append(game_name)
                        claims['pure_ne_identical']['supported'] = True
                        break

        # Claim 4: Mixed NE different
        if result['classical_mixed'] and result['pt_ne']:
            p_class, q_class = result['classical_mixed']
            p_pt, q_pt = result['pt_ne']
            delta = abs(p_class - p_pt) + abs(q_class - q_pt)
            if delta > 0.1:  # Significant difference
                claims['mixed_ne_different']['games'].append(game_name)
                claims['mixed_ne_different']['supported'] = True

    return claims
