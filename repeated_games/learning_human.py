import torch
import torch.nn as nn
import torch.optim as optim
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

        # Initialize belief lambda parameter (subject to tuning)
        self.lam = 0.9
       
        # Add an entry for each state populated with uniform probabilities over opponent action set size
        # And initialize q values
        for state in range(self.state_size):
            # Belief function
            self.beliefs[state] = torch.ones(self.opp_action_size) / self.opp_action_size
            
            # Q-values
            self.q_values[state] = torch.zeros(self.action_size, self.opp_action_size) 

        # Q-learning parameters
        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.001

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

    ###### DOUBLE CHECK THE INPUT TO THE STATE ARGUMENT, NEED A NUMBER TO ACCOUNT FOR INDEXING ###
    def act(self, state, training=True):
        """Choose action using epsilon-greedy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0) # CCOME BACK TO THIS VARIABLE

        # Epsilon exploration (lines 16-17 in alg 1)
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Pathology detection (lines 14, 18, 19 in alg 1)
        action_values = self.calculate_action_values(state)

        ## Identify Optimal action
        optimal_action = torch.argmax(action_values) 
        
        ## Identify second best action
        non_optimal_actions = action_values.clone()
        non_optimal_actions[optimal_action] = -torch.inf
        second_best_action = non_optimal_actions.max()

        ## Check for pathology:
        if abs(action_values[optimal_action] - second_best_action) < self.tau:
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

            outcomes = self.q_values[state][action]

            action_val = self.pt.expected_pt_value(outcomes, probabilities) 
            action_values[action] = action_val

        return action_values

    def belief_update(self, state, opp_action):
        one_hot = torch.zeros(self.opp_action_size)
        one_hot[opp_action] = 1
        self.beliefs[state] = self.lam * self.beliefs[state] + (1 - self.lam) * one_hot

    def q_value_update(self, state, next_state, action, opp_action, reward):
        # Get optimal action (tensor of length self.action_size)
        action_values = self.calculate_action_values(next_state)
        # Get maximuj value (not index)
        optimal_value = action_values.max()
        # Get stored value (state, joint action value) 
        q_value = self.q_values[state][action][opp_action]

        # Calculate delta in pt value space
        delta = self.transform_reward(reward) + self.gamma * optimal_value - q_value 
        # Update q values
        self.q_values[state][action][opp_action] += self.alpha * delta




