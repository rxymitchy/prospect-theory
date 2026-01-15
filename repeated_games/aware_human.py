from .ProspectTheory import ProspectTheory
import numpy as np
import random
from scipy.special import softmax

class AwareHumanPTAgent:
    """
    Sophisticated Aware Human PT Agent
    Knows the game structure and uses PT to compute best responses
    """

    def __init__(self, payoff_matrix, pt_params, action_size, state_size, agent_id=0, opp_params=None):
        self.payoff_matrix = payoff_matrix
        self.pt = ProspectTheory(**pt_params)
        self.agent_id = agent_id  # 0 for row player, 1 for column player
        self.learning_rate = 0.1
        self.ref_point = 0 # amenable to changes if we want to test different ref points
        self.tau = 0.1 # Also amenable
        self.temperature = 1.3

        self.action_size = action_size
        self.opp_action_size = opp_params['opponent_action_size']
        self.state_size = state_size

        self.values = self.build_values_matrix(self.payoff_matrix) 

        self.opponent_type = opp_params['opponent_type']

        if self.opponent_type == "AI":
            # Record value table
            self.opp_q_values = opp_params['q_values']
            # epsilon greedy formula
            self.opp_epsilon = opp_params['epsilon']
        elif self.opponent_type == "LH":
            # Record value table
            self.opp_q_values = opp_params['q_values']
            # Record epsilon
            self.opp_epsilon = opp_params['epsilon']
            # Record beliefs 
            self.opp_beliefs = opp_params['beliefs']

    def build_values_matrix(self, matrix):
        values = np.zeros((self.action_size, self.opp_action_size))

        for i in range(self.action_size):
            for j in range(self.opp_action_size):
                values[i, j] = matrix[i, j, self.agent_id]

        return values
                

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
        if setting == "direct_action" and opponent_action is not None:
            self.opponent_action = opponent_action

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

    def calculate_opponent_policy(self, state):
        """Taking in the opponent's decision parameters, returns a 
           CPT transformation of self.values based on likelihood that 
           the opponent takes action a or b."""
        # AI agent
        if self.opponent_type == "AI":
            probs = np.zeros_like(self.opp_q_values[state])

            # Get idx of optimal opp action
            optimal_action_idx = np.argmax(self.opp_q_values[state])

            # 1 - opp_epsilon/opp_action_size because the highest value action :=
            # (p(argmax) = 1 - opp_epsilon) + (p(random) = opp_epsilon/opp_action_size) 
            probs[optimal_action_idx] = 1 - self.opp_epsilon + self.opp_epsilon / self.opp_action_size

            for idx in range(len(probs)):
                if idx == optimal_action_idx:
                    continue
                else:
                    probs[idx] = self.opp_epsilon/self.opp_action_size
                    
            # Now use probability of opponent actions to redefine our payoffs:
            transformed_payoffs = np.zeros(self.action_size)
            # Form a prospect for each (action, opp_action_prob)
            for a_i in range(self.action_size):
                if self.agent_id == 0: #row player
                     transformed_value = self.pt.expected_pt_value(self.values[a_i, :], probs)
                     transformed_payoffs[a_i] = transformed_value
                else: # column player
                     transformed_value = self.pt.expected_pt_value(self.values[:, a_i], probs)
                     transformed_payoffs[a_i] = transformed_value
            
            return transformed_rewards

    def act(self):
        """Choose action based on PT analysis"""
        # Get opp strategy
        if opp_action is None:
            raise ValueError("opponent_action not set; call retrieve_opponent_strategy first.")

        opp_action = self.opponent_action

        # Compute optimal action
        action = self.compute_best_response(opp_action)

        return action

