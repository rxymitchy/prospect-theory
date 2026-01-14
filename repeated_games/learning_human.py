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

        # Initialize beliefs function and q values as dictionaries
        self.beliefs = dict()
        self.q_values = dict()
       
        # Add an entry for each state populated with uniform probabilities over opponent action set size
        # And initialize q values
        for state in range(self.state_size):
            # Belief function
            self.beliefs[state] = torch.ones(self.opp_action_size) / self.opp_action_size
            
            # Q-values
            self.q_values[state] = torch.zeros(self.actions_size, self.opp_action_size) 

        # Q-learning parameters
        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Pathology Detection parameters
        self.tau = 0.1
        ## subject to change
        self.temperature = 1.3 

        #self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        #self.criterion = nn.MSELoss()

        # Experience replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32

        # Track raw vs PT rewards
        self.raw_rewards = []
        self.pt_rewards = []

    def transform_reward(self, reward):
        """Transform raw reward through PT value function"""
        pt_reward = self.pt.value_function(reward - self.pt.r)
        self.raw_rewards.append(reward)
        self.pt_rewards.append(pt_reward)
        return pt_reward

    def remember(self, state, action, opp_action, reward, next_state, done):
        """Store experience with PT-transformed reward"""
        # Come back to this: still not sure that we should be storing values and not outcomes
        pt_reward = self.transform_reward(reward)
        self.memory.append((state, action, opp_action, pt_reward, next_state, done))

    ###### DOUBLE CHECK THE INPUT TO THE STATE ARGUMENT, NEED A NUMBER TO ACCOUNT FOR INDEXING ###
    def act(self, state, training=True):
        """Choose action using epsilon-greedy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5), \
        "Beliefs don't sum to 1"
        # Epsilon exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Pathology detection
        action_values = torch.zeros(self.action_size)
        ## Calculate V for each state, joint action tuple
        for action in range(self.action_size):
            probabilities = self.beliefs[state] 
            assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5), \
            "Beliefs don't sum to 1"

            outcomes = self.q_values[state][action]

            action_val = self.pt.expected_pt_value(outcomes, probabilities) 
            action_values[action] = action_val

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
        
        # Optimal Action
        else:
            return int(optimal_action.item())
             
        
  



    def replay(self):
        """Train on experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Is random correct here?
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q = self.q_net(states).gather(1, actions).squeeze()

        # Next Q values
        with torch.no_grad():
            next_q = self.q_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Train
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self, my_action, opponent_action, reward=None):
        """Update agent (for compatibility)"""
        pass

    def get_pt_stats(self):
        """Get statistics about PT transformation"""
        if not self.raw_rewards:
            return {"mean_raw": 0, "mean_pt": 0, "std_raw": 0, "std_pt": 0}

        return {
            "mean_raw": np.mean(self.raw_rewards[-100:]),
            "mean_pt": np.mean(self.pt_rewards[-100:]),
            "std_raw": np.std(self.raw_rewards[-100:]),
            "std_pt": np.std(self.pt_rewards[-100:])
        }

