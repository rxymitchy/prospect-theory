# ============================================================================
# 2. GAME DEFINITIONS (expanded set)
# ============================================================================

def get_all_games():
    """Return dictionary of all games for testing"""

    games = {
        # ============================================================
        # ORIGINAL GAMES
        # ============================================================
        'PrisonersDilemma': {
            'payoffs': np.array([
                [[-1, -1], [-3, 0]],   # Cooperate/Cooperate, Cooperate/Defect
                [[0, -3], [-2, -2]]    # Defect/Cooperate, Defect/Defect
            ]),
            'description': 'Classic Prisoner\'s Dilemma',
            'classical_NE': {
                'pure': [(1, 1)],  # Both defect (index 1 = Defect)
                'mixed': None
            }
        },

        'MatchingPennies': {
            'payoffs': np.array([
                [[1, -1], [-1, 1]],   # Heads/Heads, Heads/Tails
                [[-1, 1], [1, -1]]    # Tails/Heads, Tails/Tails
            ]),
            'description': 'Zero-sum matching pennies',
            'classical_NE': {
                'pure': [],
                'mixed': (0.5, 0.5)
            }
        },

        'OchsGame_Correct': {
            'payoffs': np.array([
                [[0, 0], [5, 8]],    # Top/Left, Top/Right
                [[8, 5], [0, 0]]     # Bottom/Left, Bottom/Right
            ]),
            'description': 'Ochs game that causes PT-NE pathology',
            'classical_NE': {
                'pure': [],
                'mixed': (0.2, 0.5)  # p=0.2, q=0.5
            }
        },

        'CrawfordGame': {
            'payoffs': np.array([
                [[0, 0], [5, 1]],    # Top/Left, Top/Right
                [[1, 5], [4, 4]]     # Bottom/Left, Bottom/Right
            ]),
            'description': 'Crawford game with convex preferences',
            'classical_NE': {
                'pure': [(0, 1), (1, 0)],  # (Top,Right) and (Bottom,Left)
                'mixed': (0.833, 0.833)    # p=5/6 ≈ 0.833
            }
        },

        'BattleOfSexes': {
            'payoffs': np.array([
                [[2, 1], [0, 0]],    # Opera/Opera, Opera/Football
                [[0, 0], [1, 2]]     # Football/Opera, Football/Football
            ]),
            'description': 'Classic Battle of the Sexes',
            'classical_NE': {
                'pure': [(0, 0), (1, 1)],
                'mixed': (2/3, 1/3)  # p=2/3, q=1/3
            }
        },

        # ============================================================
        # NEW GAMES FOR COMPREHENSIVE TESTING
        # ============================================================
        'StagHunt': {
            'payoffs': np.array([
                [[4, 4], [0, 3]],    # Stag/Stag, Stag/Hare
                [[3, 0], [2, 2]]     # Hare/Stag, Hare/Hare
            ]),
            'description': 'Stag Hunt - coordination with risk',
            'classical_NE': {
                'pure': [(0, 0), (1, 1)],  # Both Stag or both Hare
                'mixed': (0.5, 0.5)
            }
        },

        'HawkDove': {
            'payoffs': np.array([
                [[0, 0], [4, 1]],    # Hawk/Hawk, Hawk/Dove
                [[1, 4], [2, 2]]     # Dove/Hawk, Dove/Dove
            ]),
            'description': 'Hawk-Dove / Chicken game',
            'classical_NE': {
                'pure': [(0, 1), (1, 0)],  # One Hawk, one Dove
                'mixed': (0.5, 0.5)
            }
        },

        'AsymmetricBoS': {
            'payoffs': np.array([
                [[3, 2], [0, 0]],    # Opera/Opera, Opera/Football
                [[0, 0], [2, 3]]     # Football/Opera, Football/Football
            ]),
            'description': 'Asymmetric Battle of Sexes',
            'classical_NE': {
                'pure': [(0, 0), (1, 1)],
                'mixed': (0.6, 0.4)  # p=0.6 Opera, q=0.4 Opera
            }
        },

        'PathologicalGame': {
            'payoffs': np.array([
                [[5, 5], [0, 8]],    # Top/Left, Top/Right
                [[8, 0], [1, 1]]     # Bottom/Left, Bottom/Right
            ]),
            'description': 'Designed to potentially break PT-NE',
            'classical_NE': {
                'pure': [(0, 0), (1, 1)],
                'mixed': (0.5, 0.5)
            }
        },

        'RiskShiftGame': {
            'payoffs': np.array([
                [[2, 2], [0, 3]],    # Safe/Safe, Safe/Risky
                [[3, 0], [1, 1]]     # Risky/Safe, Risky/Risky
            ]),
            'description': 'Game sensitive to reference point',
            'classical_NE': {
                'pure': [(0, 0), (1, 1)],
                'mixed': (0.5, 0.5)
            }
        },

        # ============================================================
        # ADDITIONAL TEST CASES
        # ============================================================
        'PureCoordination': {
            'payoffs': np.array([
                [[1, 1], [0, 0]],    # A/A, A/B
                [[0, 0], [1, 1]]     # B/A, B/B
            ]),
            'description': 'Pure coordination game',
            'classical_NE': {
                'pure': [(0, 0), (1, 1)],
                'mixed': (0.5, 0.5)
            }
        },

        'Deadlock': {
            'payoffs': np.array([
                [[1, 1], [3, 0]],    # Cooperate/Cooperate, Cooperate/Defect
                [[0, 3], [2, 2]]     # Defect/Cooperate, Defect/Defect
            ]),
            'description': 'Deadlock game (variant of PD)',
            'classical_NE': {
                'pure': [(1, 1)],
                'mixed': None
            }
        }
    }

    return games
