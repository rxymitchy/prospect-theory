# ============================================================================
# ADVANCED DIAGNOSTICS SECTION
# ============================================================================

def run_advanced_diagnostics():
    """Advanced diagnostics for PT equilibrium analysis"""

    print("\n" + "="*80)
    print("ADVANCED DIAGNOSTICS")
    print("="*80)

    # Import needed functions (they should be in same file)
    from scipy.optimize import fsolve

    def debug_best_response(game_name, payoff_matrix, pt):
        """Debug best response functions"""
        print(f"\nDebugging {game_name}:")
        print(f"PT params: λ={pt.lambd}, α={pt.alpha}, γ={pt.gamma}, r={pt.r}")

        # Test BR at key points
        test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        print("\nPlayer 1 BR(q):")
        for q in test_points:
            br = find_pt_best_response(q, payoff_matrix, pt, 0)
            print(f"  BR({q:.2f}) = {br:.3f}")

        print("\nPlayer 2 BR(p):")
        for p in test_points:
            br = find_pt_best_response(p, payoff_matrix, pt, 1)
            print(f"  BR({p:.2f}) = {br:.3f}")

    def analyze_alpha_effect():
        """Analyze effect of α parameter"""
        print("\n" + "-"*60)
        print("ANALYZING α PARAMETER EFFECT")
        print("-"*60)

        # Matching Pennies game
        mp_game = np.array([
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]]
        ])

        print("\nMatching Pennies with different α (λ=0, γ=1.0, r=0):")
        print("α-value | PT-NE exists? | Notes")
        print("-" * 50)

        for alpha in [0.5, 0.88, 1.0, 1.5, 2.0]:
            pt = ProspectTheory(lambd=0, alpha=alpha, gamma=1.0, r=0)
            pt_ne = find_pt_nash_equilibrium(mp_game, pt, grid_points=21)

            exists = "YES" if pt_ne else "NO"
            note = "Standard PT" if alpha == 0.88 else "Thesis" if alpha == 2.0 else ""

            print(f"α={alpha:.2f} | {exists:11} | {note}")

    # Get games
    games = get_all_games()

    # Run diagnostics
    print("\n1. MATCHING PENNIES PATHOLOGY ANALYSIS")
    mp_pt = ProspectTheory(lambd=0, alpha=2.0, gamma=1.0, r=0)
    debug_best_response('MatchingPennies', games['MatchingPennies']['payoffs'], mp_pt)

    print("\n2. OCHS GAME ANALYSIS")
    ochs_pt = ProspectTheory(lambd=0, alpha=2.0, gamma=1.0, r=1)
    debug_best_response('OchsGame', games['OchsGame_Correct']['payoffs'], ochs_pt)

    # Analyze α parameter
    analyze_alpha_effect()

    # Key insights
    print("\n" + "="*80)
    print("KEY THEORETICAL INSIGHTS")
    print("="*80)

    insights = [
        "1. α=2.0 creates CONVEX value function (unlike standard PT α=0.88)",
        "2. Convex preferences can lead to equilibrium non-existence",
        "3. Matching Pennies pathology explained by convex value function",
        "4. Thesis results are specific to α=2.0 parameterization",
        "5. With standard α=0.88, PT-NE might exist in more games"
    ]

    for insight in insights:
        print(f"• {insight}")
