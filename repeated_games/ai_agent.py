import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class AIAgent:
    """Standard RL agent without PT"""

    def __init__(self, state_size, action_size, agent_id=0):
        self.state_size = state_size
        self.action_size = action_size
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
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
        pass
