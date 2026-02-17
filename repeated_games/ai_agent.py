import torch
import random
from torch.nn import Softmax

class AIAgent:
    """Standard RL agent without PT"""

    def __init__(self, state_size, action_size, opp_action_size, agent_id=0):
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

        # state counter
        self.state_visit_counter = dict()

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
        self.softmax_counter = 0

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
            # Log tie break
            self.softmax_counter += 1

            vals = q_values - q_values.max() # Normalize to prevent explosions
            probs = self.softmax(vals / self.temp)
            action = torch.multinomial(probs, 1).item() # sample

            return action

        else:
            return optimal_action
 

    def update(self, state, action, next_state, reward=None, done=False):
        assert reward is not None, "Reward Undefined"
        
        # Update state count
        if state not in self.state_visit_counter.keys():
            self.state_visit_counter[state] = 0

        self.state_visit_counter[state] += 1

        curr_q_val = self.q_values[state][action]
        max_next_q_val = self.q_values[next_state].max()

        # get rid of future trajectories when reaching end of episode
        if done is True:
            max_next_q_val = 0

        target = reward + self.gamma * max_next_q_val
        self.q_values[state][action] = (1 - self.alpha) * curr_q_val + self.alpha * target  
       
    # Deprecated Code for the Q-Value convergence metric I was fixated on
    def get_q_values(self):
        q_values = torch.zeros(self.action_size, self.opp_action_size)

        total_visits = sum(self.state_visit_counter.values())

        if total_visits == 0:
            return q_values

        for state, q_val in self.q_values.items():
            num_visits = self.state_visit_counter.get(state, 0)

            if num_visits == 0:
                continue

            weight = num_visits / total_visits

            q_val = torch.as_tensor(q_val, dtype=torch.float32)

            q_val = q_val.unsqueeze(1).repeat(1, self.opp_action_size)

            q_values += weight * q_val

        q_values = q_values.numpy()

        return q_values
