from .ProspectTheory import ProspectTheory
import numpy as np
import random
import scipy.special import softmax

class AwareHumanPTAgent:
    """
    Sophisticated Aware Human PT Agent
    Knows the game structure and uses PT to compute best responses
    """

    def __init__(self, payoff_matrix, pt_params, agent_id=0, action_size, opponent_action_size, state_size):
        self.payoff_matrix = payoff_matrix
        self.pt = ProspectTheory(**pt_params)
        self.agent_id = agent_id  # 0 for row player, 1 for column player
        self.opponent_strategy = None
        self.learning_rate = 0.1
        self.ref_point = 0 # amenable to changes if we want to test different ref points
        self.tau = 0.1 # Also amenable
        self.temperature = 1.3

        self.action_size = action_size
        self.opp_action_size = opponent_action_size
        self.state_size = state_size

        # Compute PT value matrix once
        self.pt_matrix = self._compute_pt_matrix()

    def _compute_pt_matrix(self):
        """Compute PT values for all outcomes"""
        pt_matrix = np.zeros_like(self.payoff_matrix)
        for i in range(self.action_size):
            for j in range(self.opp_action_size):
                for player in [0, 1]:
                    payoff = self.payoff_matrix[i, j, player]
                    pt_matrix[i, j, player] = self.pt.value_function(payoff - self.ref_point)
        return pt_matrix

    def retrieve_opponent_strategy(self, opponent_action=None, opponent_probs=None, opponent_policy=None):
        """ Its unclear right now what the best modeling choice is for the aware human. We have several options.
            1) directly pass the opponent's action, 2) directly pass the opponent's mixed strategy, 3) directly 
            pass the opponent's policy. Each have their own modeling tradeoffs: direct action is simple, effective, 
            and easy to implement, but what humans in the world KNOW what the action they will face will be? Direct 
            probabilities is a bit like beliefs that have epistemic grounding, which is somewhat similar to LH and its 
            not completely clear where we would generate these mixed strategies (from the literature, sure, but I suspect
            EU will be easier than PT for this task). Direct policy probably makes more sense, but it would require 
            adaptation for each agent pairing and so its a more difficult implementation with more shuffling. Here, for now,
            I'm going to implement direct action, but please be aware of the alternatives. """ 

        # modeling strategy
        setting = "direct_action"
        if setting == "direct_action" and opponent_action:
            self.opponent_strategy = opponent_action

    def compute_best_response(self, opp_action):
        """
        Compute PT best response to a known opponent's action
        NOTE (1/14/26): I rewrote this with the assumption that the opponent's action will
        be directly given to the agent, so I removed uncertainty and full pt value calculation
        """
        # Expected PT values for each of our actions
        expected_values = []

        for my_action in range(self.action_size):
            # Get PT value for this outcome
            if self.agent_id == 0:  # Row player
                pt_value = self.pt_matrix[my_action, opp_action, 0]
            else:  # Column player
                pt_value = self.pt_matrix[opp_action, my_action, 1]

            expected_values.append(pt_value)

        # Make expected values an np array for operations
        expected_values = np.array(expected_values)

        # Argmax response based on PT values
        optimal_action = np.argmax(expected_values)
        suboptimal_actions = expected_values[:]
        suboptimal_actions[optimal_action] = -np.inf
        second_best = np.argmax(suboptimal_actions)

        # No clear optimum detection
        gap = expected_values[optimal_action] - expected_values[second_best]
        if gap < self.tau:
            logits = expected_values - expected_values.max()
            probs = softmax(logits / self.temperature, axis=0)
            action = np.random.choice(self.action_size, p=probs)
        else:
            action = optimal_action

        return action

    def act(self, training=True):
        """Choose action based on PT analysis"""
        # Estimate opponent strategy
        opp_strat = self.opponent_strategy

        # Compute optimal probability for action 0
        action = self.compute_best_response(opp_strat)

        return action

