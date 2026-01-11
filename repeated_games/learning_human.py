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

    def __init__(self, state_size, action_size, pt_params, agent_id=0):
        self.state_size = state_size
        self.action_size = action_size
        self.pt = ProspectTheory(**pt_params)
        self.agent_id = agent_id

        # Q-learning parameters
        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Neural network
        self.q_net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

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

    def remember(self, state, action, reward, next_state, done):
        """Store experience with PT-transformed reward"""
        pt_reward = self.transform_reward(reward)
        self.memory.append((state, action, pt_reward, next_state, done))

    def act(self, state, training=True):
        """Choose action using epsilon-greedy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        """Train on experiences"""
        if len(self.memory) < self.batch_size:
            return

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

