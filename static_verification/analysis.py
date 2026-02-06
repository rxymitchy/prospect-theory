# ============================================================================
# 5. MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_game(game_name, game_data, pt_params):
    """
    Analyze a single game with given PT parameters

    Parameters:
    -----------
    game_name : str, name of the game
    game_data : dict, game definition
    pt_params : dict, PT parameters

    Returns:
    --------
    dict: analysis results
    """
    payoff_matrix = game_data['payoffs']
    pt = ProspectTheory(**pt_params)

    # Find classical NE
    pure_ne, mixed_ne = find_classical_ne(payoff_matrix)

    # Find PT-NE
    pt_ne = find_pt_nash_equilibrium(payoff_matrix, pt, grid_points=31)

    # Find PT-EB
    pt_eb = find_pt_equilibrium_in_beliefs(payoff_matrix, pt, grid_points=41)

    # Compare with classical
    pt_vs_classical = "Unknown"
    if mixed_ne and pt_ne:
        p_class, q_class = mixed_ne
        p_pt, q_pt = pt_ne
        delta = abs(p_class - p_pt) + abs(q_class - q_pt)
        pt_vs_classical = "Similar" if delta < 0.1 else "Different"

    return {
        'game': game_name,
        'description': game_data['description'],
        'classical_pure': pure_ne,
        'classical_mixed': mixed_ne,
        'pt_ne': pt_ne,
        'pt_eb': pt_eb,
        'pt_vs_classical': pt_vs_classical,
        'pt_params': pt_params.copy()
    }

def run_comprehensive_analysis():
    """Run comprehensive analysis on all games"""

    # Get all games
    games = get_all_games()

    # Define parameter sets to test
    parameter_sets = [
        # Baseline (standard PT from literature)
        {'lambd': 2.25, 'alpha': 0.88, 'gamma': 0.61, 'r': 0, 'name': 'Baseline'},

        # From thesis examples
        {'lambd': 2.25, 'alpha': 2.0, 'gamma': 1.0, 'r': 0, 'name': 'PrisonersDilemma'},
        {'lambd': 0, 'alpha': 2.0, 'gamma': 1.0, 'r': 0, 'name': 'MatchingPennies'},
        {'lambd': 0, 'alpha': 2.0, 'gamma': 1.0, 'r': 1, 'name': 'OchsGame'},
        {'lambd': 0, 'alpha': 2.0, 'gamma': 1.0, 'r': 0, 'name': 'CrawfordGame'},
        {'lambd': 1.0, 'alpha': 2.0, 'gamma': 1.0, 'r': 0, 'name': 'BattleOfSexes'},

        # Additional tests
        {'lambd': 0, 'alpha': 1.0, 'gamma': 1.0, 'r': 0, 'name': 'NoLossNoWeight'},
        {'lambd': 1.5, 'alpha': 0.88, 'gamma': 0.61, 'r': 0, 'name': 'ModerateLoss'},
        {'lambd': 2.25, 'alpha': 0.88, 'gamma': 0.8, 'r': 0, 'name': 'WeakWeighting'},
        {'lambd': 2.25, 'alpha': 0.88, 'gamma': 0.4, 'r': 0, 'name': 'StrongWeighting'},
        {'lambd': 2.25, 'alpha': 0.88, 'gamma': 0.61, 'r': 1.0, 'name': 'RefPoint1'},
    ]

    # Game-parameter combinations to test (focused)
    test_combinations = [
        # Thesis core examples
        ('PrisonersDilemma', 'PrisonersDilemma'),
        ('MatchingPennies', 'MatchingPennies'),
        ('OchsGame_Correct', 'OchsGame'),
        ('CrawfordGame', 'CrawfordGame'),
        ('BattleOfSexes', 'BattleOfSexes'),

        # Additional tests
        ('StagHunt', 'Baseline'),
        ('HawkDove', 'Baseline'),
        ('AsymmetricBoS', 'Baseline'),
        ('PathologicalGame', 'Baseline'),
    ]

    results = []

    for game_name, param_name in test_combinations:
        # Find parameter set
        params = next(p for p in parameter_sets if p['name'] == param_name)
        param_dict = {k: v for k, v in params.items() if k != 'name'}

        # Analyze game
        result = analyze_game(game_name, games[game_name], param_dict)
        results.append(result)

    return results
