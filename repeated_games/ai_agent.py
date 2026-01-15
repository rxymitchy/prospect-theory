import torch
import random

class AIAgent:
    """Standard RL agent without PT"""

    def __init__(self, state_size, action_size, pt_params, agent_id=0):
        self.state_size = state_size
        self.action_size = action_size
        self.opp_action_size = opp_action_size
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
        self.alpha = 0.001

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        else:
            action = torch.argmax(self.q_values[state]).item() 
            return action

    def update(self, state, action, next_state, reward=None):
        assert reward is not None, "Reward Undefined"
        curr_q_val = self.q_values[state][action]
        max_next_q_val = self.q_values[next_state].max()
        target = reward + self.gamma * max_next_q_val
        self.q_values[state][action] = (1 - self.alpha) * curr_q_val + self.alpha * target  
