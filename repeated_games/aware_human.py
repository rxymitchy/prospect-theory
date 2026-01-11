from .ProspectTheory import ProspectTheory
import numpy as np
import random

class AwareHumanPTAgent:
    """
    Sophisticated Aware Human PT Agent
    Knows the game structure and uses PT to compute best responses
    """

    def __init__(self, payoff_matrix, pt_params, agent_id=0):
        self.payoff_matrix = payoff_matrix
        self.pt = ProspectTheory(**pt_params)
        self.agent_id = agent_id  # 0 for row player, 1 for column player
        self.opponent_history = []
        self.my_history = []
        self.learning_rate = 0.1

        # Compute PT value matrix once
        self.pt_matrix = self._compute_pt_matrix()

        # Track opponent strategy estimate
        self.opponent_strategy_est = 0.5  # Initially assume uniform

    def _compute_pt_matrix(self):
        """Compute PT values for all outcomes"""
        pt_matrix = np.zeros_like(self.payoff_matrix)
        for i in range(2):
            for j in range(2):
                for player in [0, 1]:
                    payoff = self.payoff_matrix[i, j, player]
                    pt_matrix[i, j, player] = self.pt.value_function(payoff - self.pt.r)
        return pt_matrix

    def estimate_opponent_strategy(self, window=50):
        """Estimate opponent's mixed strategy using Bayesian updating"""
        if len(self.opponent_history) < 10:
            return 0.5  # Insufficient data

        # Use recent history
        recent = self.opponent_history[-window:] if window else self.opponent_history
        count_0 = sum(1 for a in recent if a == 0)
        total = len(recent)

        # Bayesian estimate with Beta(1,1) prior
        alpha = 1 + count_0
        beta = 1 + (total - count_0)
        return alpha / (alpha + beta)

    def compute_best_response(self, opponent_strat):
        """
        Compute PT best response to opponent's strategy
        Returns probability of playing action 0
        """
        # Expected PT values for each of our actions
        expected_values = []

        for my_action in [0, 1]:
            # Possible opponent actions
            outcomes = []
            probabilities = []

            for opp_action in [0, 1]:
                # Get PT value for this outcome
                if self.agent_id == 0:  # Row player
                    pt_value = self.pt_matrix[my_action, opp_action, 0]
                else:  # Column player
                    pt_value = self.pt_matrix[opp_action, my_action, 1]

                # Probability of opponent action
                prob = opponent_strat if opp_action == 0 else (1 - opponent_strat)

                outcomes.append(pt_value)
                probabilities.append(prob)

            # Calculate PT expected value
            exp_val = self.pt.expected_pt_value(np.array(outcomes), np.array(probabilities))
            expected_values.append(exp_val)

        # Softmax response based on PT values
        exp_values = np.array(expected_values)
        exp_values = np.exp(exp_values - np.max(exp_values))  # Numerical stability
        probabilities = exp_values / np.sum(exp_values)

        return probabilities[0]  # Probability of playing action 0

    def act(self, training=True, exploration_rate=0.1):
        """Choose action based on PT analysis"""
        # Estimate opponent strategy
        opp_strat = self.estimate_opponent_strategy()

        # Compute optimal probability for action 0
        p_optimal = self.compute_best_response(opp_strat)

        # Exploration during training
        if training and random.random() < exploration_rate:
            action = random.choice([0, 1])
        else:
            # Stochastic best response
            action = 0 if random.random() < p_optimal else 1

        return action

    def update(self, my_action, opponent_action, reward=None):
        """Update history after playing"""
        self.my_history.append(my_action)
        self.opponent_history.append(opponent_action)

        # Update strategy estimate (exponential smoothing)
        if len(self.opponent_history) > 1:
            recent_opp_0 = sum(1 for a in self.opponent_history[-10:] if a == 0) / min(10, len(self.opponent_history))
            self.opponent_strategy_est = (1 - self.learning_rate) * self.opponent_strategy_est + \
                                         self.learning_rate * recent_opp_0

    def get_strategy(self):
        """Get current strategy estimate"""
        opp_strat = self.estimate_opponent_strategy()
        p_optimal = self.compute_best_response(opp_strat)
        return p_optimal
