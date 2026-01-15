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
            # Decision parameters:
            self.opp_tau = opp_params['tau']
            self.opp_temp = opp_params['temp']

    def build_values_matrix(self, matrix):
        values = np.zeros((self.action_size, self.opp_action_size))

        for i in range(self.action_size):
            for j in range(self.opp_action_size):
                values[i, j] = matrix[i, j, self.agent_id]

        return values

    def calculate_opponent_policy(self, state):
        """Taking in the opponent's decision parameters, returns a 
           CPT transformation of self.values based on likelihood that 
           the opponent takes action a or b."""
        # AI agent
        if self.opponent_type == "AI":
            probs = np.zeros(self.opp_action_size)

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

        elif self.opponent_type == "LH":
            # Action Probabilities Initialize
            probs = np.zeros(self.opp_action_size)
            # CPT Transformation between beliefs and q values
            action_values = np.zeros(self.opp_action_size)
            belief_probs = self.opp_beliefs[state]

            assert np.isclose(belief_probs.sum(), 1.0, atol=1e-5), \
            "Beliefs don't sum to 1"
            
            # Compute LH policy inside model. The point is that we need to know if the gap 
            # (pathology detection) exists 
            ################# BEGIN LH POLICY CALCULATION ###################
            for action in range(self.opp_action_size):
                opp_values = self.opp_q_values[state][action]
                action_cpt_value = self.pt.expected_pt_value(opp_values, belief_probs)
                action_values[action] = action_cpt_value

            optimal_action = np.argmax(action_values)
            suboptimal_action_values = action_values[:]
            suboptimal_action_values[optimal_action] = -np.inf

            second_best_action = np.argmax(suboptimal_action_values)

            gap = action_values[optimal_action] - action_values[second_best_action]
            soft_probs = None
            if gap < self.opp_tau:
                vals = action_values - action_values.max()
                soft_probs = softmax(vals / self.opp_temp, axis=0)
            ############### END LH POLICY CALCULATION ####################

            # Compute and store action probs
            # Use opp action probs to make AH action decision.
            if soft_probs is not None:
                # The gap is triggered, so we use the soft probs
                # HOWEVER, the epsilon greedy logic occurs before this step,
                # SO we need to account for the likelihood that epsilon triggers
                # Formally: p(epsilon) + p(not epsilon | softmax)
                # IMPORTANTLY, because we know that softmax is triggered because
                # we calculated the policy directly, this is complete. 
                # sum(uniform) = 1 and sum(soft_probs) = 1, so the epsilon scaling still results in 1. 
                uniform = np.ones(self.opp_action_size) / self.opp_action_size
                probs = self.opp_epsilon * uniform + (1 - self.opp_epsilon) * soft_probs
                assert np.isclose(probs.sum(), 1.0, atol=1e-5), \
                "Probs don't sum to 1"
            else:
                probs[optimal_action] = 1 - self.opp_epsilon + self.opp_epsilon / self.opp_action_size
                for idx in range(self.opp_action_size):
                    if idx == optimal_action: 
                        continue
                    else:
                        probs[idx] = self.opp_epsilon / self.opp_action_size 

        return probs
 
    def cpt_transform(self, probs):
        """
            Input: Opponent strategy calculated in calculate_opponent_policy
            Output: CPT transformation where a prospect is P(Payoff, opp_strat)
        """
        # Now use probability of opponent actions to redefine our (AH) payoffs:
        transformed_payoffs = np.zeros(self.action_size)
        # Form a prospect for each (action, opp_action_prob)
        for a_i in range(self.action_size):
            if self.agent_id == 0: #row player
                transformed_value = self.pt.expected_pt_value(self.values[a_i, :], probs)
                transformed_payoffs[a_i] = transformed_value
            else: # column player
                transformed_value = self.pt.expected_pt_value(self.values[:, a_i], probs)
                transformed_payoffs[a_i] = transformed_value
    
        return transformed_payoffs


    def act(self, state):
        """
           Calculate opponent policy, CPT transform own values with opponent action probs, 
           pathology detection or argmax (no exploration)
        """
        opp_strat = self.calculate_opponent_policy(state)
        transformed_payoffs = self.cpt_transform(opp_strat)

        # Compute optimal action
        optimal_action = np.argmax(transformed_payoffs)
        suboptimal_actions = transformed_payoffs[:]
        suboptimal_actions[optimal_action] = -np.inf
        second_best_action = np.argmax(suboptimal_actions)

        gap = transformed_payoffs[optimal_action] - transformed_payoffs[second_best_action]

        if gap < self.tau:
            vals = transformed_payoffs - transformed_payoffs.max()
            probs = softmax(vals / self.temperature, axis=0)
            action = np.random.choice(self.action_size, p=probs)

        else:
            action = optimal_action

        return action

