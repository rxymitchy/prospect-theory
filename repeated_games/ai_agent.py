import torch
import random
from torch.nn import Softmax

class AIAgent:
    """Standard RL agent without PT"""

    def __init__(self, state_size, action_size, agent_id=0):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id

        # Initialize q values as a dictionary
        self.q_values = dict()
       
        # Add an entry for each state 
        # And initialize q values
        for state in range(self.state_size):
            self.q_values[state] = torch.zeros(self.action_size) 

        # Q-learning parameters
        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.01

        # tiebreaker
        self.tau = 0.1
        self.temp = 0.7
        self.softmax = Softmax(dim=0)

    def act(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.q_values[state]
        optimal_action = torch.argmax(q_values).item() 

        suboptimal_q_values = q_values.clone()
        suboptimal_q_values[optimal_action] = -torch.inf
        second_best_action = torch.argmax(suboptimal_q_values).item()

        gap = q_values[optimal_action] - q_values[second_best_action]

        if gap < self.tau:
            print('[Debug AI] Softmax')
            vals = q_values - q_values.max() # Normalize to prevent explosions
            probs = self.softmax(vals / self.temp)
            action = torch.multinomial(probs, 1).item() # sample

            return action

        else:
            return optimal_action
 

    def update(self, state, action, next_state, reward=None):
        assert reward is not None, "Reward Undefined"
        curr_q_val = self.q_values[state][action]
        max_next_q_val = self.q_values[next_state].max()
        target = reward + self.gamma * max_next_q_val
        self.q_values[state][action] = (1 - self.alpha) * curr_q_val + self.alpha * target  
