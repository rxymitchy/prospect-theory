import numpy as np

def get_all_games():
    """Return all games for testing"""
    return {
        'PrisonersDilemma': {
            'payoffs': np.array([
                [[-1, -1], [-3, 0]],   # C/C, C/D
                [[0, -3], [-2, -2]]    # D/C, D/D
            ]),
            'description': 'Classic Prisoner\'s Dilemma',
            'actions': ['Cooperate', 'Defect']
        },
        'MatchingPennies': {
            'payoffs': np.array([
                [[1, 0], [0, 1]],   # H/H, H/T
                [[0, 1], [1, 0]]    # T/H, T/T
            ]),
            'description': 'Zero-sum Matching Pennies',
            'actions': ['Heads', 'Tails']
        },
        'BattleOfSexes': {
            'payoffs': np.array([
                [[2, 1], [0, 0]],    # Opera/Opera, Opera/Football
                [[0, 0], [1, 2]]     # Football/Opera, Football/Football
            ]),
            'description': 'Battle of the Sexes',
            'actions': ['Opera', 'Football']
        },
        'StagHunt': {
            'payoffs': np.array([
                [[4, 4], [0, 3]],    # Stag/Stag, Stag/Hare
                [[3, 0], [2, 2]]     # Hare/Stag, Hare/Hare
            ]),
            'description': 'Stag Hunt Coordination Game',
            'actions': ['Stag', 'Hare']
        },
        'Chicken': {
            'payoffs': np.array([
                [[0, 0], [-1, 1]],   # Swerve/Swerve, Swerve/Straight
                [[1, -1], [-10, -10]] # Straight/Swerve, Straight/Straight
            ]),
            'description': 'Chicken Game',
            'actions': ['Swerve', 'Straight']
        },
        'OchsGame': {
            'payoffs': np.array([
                [[4, 0], [0, 1]],    # Top/Left, Top/Right
                [[0, 1], [1, 0]]     # Bottom/Left, Bottom/Right
            ]),
            'description': 'Ochs Game - PT equilibrium pathology',
            'actions': ['Top', 'Bottom']
        },
        'CrawfordGame': {
            'payoffs': np.array([
                [[1, -1], [0, 0]],    # Top/Left, Top/Right
                [[0, 0], [-1, 1]]     # Bottom/Left, Bottom/Right
            ]),
            'description': 'Crawford Game - convex preference effects',
            'actions': ['Top', 'Bottom']
        }
}

def best_responders(agent1, agent2, agent1_type, agent2_type, action_size, payoff_matrix, pt_params, env, ref_setting, ref_lambda):
    opp_params = dict()
    opp_params['opponent_type'] = agent2_type
    opp_params['opponent_action_size'] = action_size
    opp_params['opp_ref'] = None
    if agent2_type != "AI":
        opp_params['opp_ref'] = agent2.ref_point
    best_responder1 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=0, opp_params=opp_params,ref_setting=ref_setting, lambda_ref=ref_lambda)

    opp_params = dict()
    opp_params['opponent_type'] = agent1_type
    opp_params['opponent_action_size'] = action_size
    opp_params['opp_ref'] = None
    if agent1_type != "AI":
        opp_params['opp_ref'] = agent1.ref_point
    best_responder2 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=1, opp_params=opp_params, ref_setting=ref_setting,                      lambda_ref=ref_lambda)

    return best_responder1, best_responder2

def smooth(y, window):
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode='valid')
