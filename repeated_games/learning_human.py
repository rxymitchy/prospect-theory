import torch
from .ProspectTheory import ProspectTheory
import numpy as np
import random

class LearningHumanPTAgent:
    """
    Learning Human PT Agent
    Doesn't know game structure, learns via RL
    Transforms rewards through PT
    """

    def __init__(self, state_size, action_size, opp_action_size, pt_params, agent_id=0):
        self.state_size = state_size
        self.action_size = action_size
        self.opp_action_size = opp_action_size
        self.pt = ProspectTheory(**pt_params)
        self.agent_id = agent_id
        self.ref_point = 0 # Come back to this for more intensive reference point testing

        # Initialize beliefs function and q values as dictionaries
        self.beliefs = dict()
        self.q_values = dict()

        # Initialize belief and reference point lambda parameters (subject to tuning)
        self.lam_b = 0.95
        self.lam_r = 0.99

        # Set reference point update mode:
        self.ref_update_mode = "EMA" # Alternate options: fixed, Q^{EU}
       
        # Add an entry for each state populated with uniform probabilities over opponent action set size
        # And initialize q values
        for state in range(self.state_size):
            # Belief function
            self.beliefs[state] = torch.ones(self.opp_action_size) / self.opp_action_size
            
            # Q-values
            self.q_values[state] = torch.zeros(self.action_size, self.opp_action_size) 

        self.state_visit_counter = dict()

        # Q-learning parameters
        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.05

        # Pathology Detection parameters
        self.tau = 0.1
        ## subject to change
        self.temperature = 1.3 

        # Track raw vs PT rewards
        self.raw_rewards = []
        self.pt_rewards = []

    # Note sure that we need this
    def transform_reward(self, reward):
        """Transform raw reward through PT value function"""
        pt_reward = self.pt.value_function(reward - self.ref_point)
        self.raw_rewards.append(reward)
        self.pt_rewards.append(pt_reward)
        return pt_reward

    def act(self, state):
        # Epsilon exploration (lines 16-17 in alg 1)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Pathology detection (lines 14, 18, 19 in alg 1)
        action_values = self.calculate_action_values(state)

        ## Identify Optimal action
        optimal_action = torch.argmax(action_values) 
        
        ## Identify second best action
        non_optimal_actions = action_values.clone()
        non_optimal_actions[optimal_action] = -torch.inf
        second_best_action = non_optimal_actions.max()

        gap = action_values[optimal_action] - second_best_action
        #print(f"[Debug] gap value LH: {gap}")

        ## Check for pathology:
        if gap < self.tau:
            print("[Debug LH] Softmax activated")
            ##Softmax
            ## Normalize 
            vals = action_values - action_values.max()
            action_probs = torch.softmax(vals/self.temperature, dim=0)
            # Sample action
            action = torch.multinomial(action_probs, 1).item()
            return action
        
        # Optimal Action (lins 20, 21 in alg 1)
        else:
            return int(optimal_action.item())

    def calculate_action_values(self, state):
        # Pathology detection (lines 14, 18, 19 in alg 1)
        action_values = torch.zeros(self.action_size)
        ## Calculate V for each state, joint action tuple
        for action in range(self.action_size):
            probabilities = self.beliefs[state] 
            assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5), \
            "Beliefs don't sum to 1"

            outcomes = self.q_values[state][action] - self.ref_point

            action_val = self.pt.expected_pt_value(outcomes, probabilities) 
            action_values[action] = action_val

        return action_values

    def belief_update(self, state, opp_action):
        one_hot = torch.zeros(self.opp_action_size)
        one_hot[opp_action] = 1
        self.beliefs[state] = self.lam_b * self.beliefs[state] + (1 - self.lam_b) * one_hot

    def ref_update(self, payoff):
        if self.ref_update_mode == "EMA":
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * payoff

    def q_value_update(self, state, next_state, action, opp_action, reward):
        if not torch.is_tensor(reward):
            reward = torch.tensor(reward, dtype=self.q_values[state].dtype)

        if state not in self.state_visit_counter.keys():
            self.state_visit_counter[state] = 0

        self.state_visit_counter[state] += 1

        # Get maximuj value (not index)
        ## - inf because rewards can be negative
        optimal_next_q_value = -torch.inf
        ## next state is necessary for double auction game
        q_values = self.q_values[next_state]
        beliefs = self.beliefs[next_state]
        for a_prime in range(self.action_size):
            q_val = q_values[a_prime]
            # linear expectation of beliefs and values
            weighted_q_val = torch.dot(beliefs, q_val).item()
            # We are maximizing
            if weighted_q_val > optimal_next_q_value:
                optimal_next_q_value = weighted_q_val

        # Get stored value (state, joint action value) 
        q_value = self.q_values[state][action][opp_action]

        # Calculate delta in untransformed reward space
        delta = reward + self.gamma * optimal_next_q_value - q_value 
        # Update q values
        self.q_values[state][action][opp_action] += self.alpha * delta

    def get_q_values(self):
        q_values = torch.zeros(self.action_size, self.opp_action_size)

        total_visits = sum(self.state_visit_counter.values())

        if total_visits == 0:
            return q_values

        for state, q_vals in self.q_values.items():
            num_visits = self.state_visit_counter.get(state, 0)

            if num_visits == 0:
                continue

            weight = num_visits / total_visits

            q_values += weight * torch.as_tensor(q_vals, dtype=torch.float32)

        return q_values.numpy()




