import numpy as np

class RepeatedGameEnv:
    """Environment for repeated 2x2 games"""

    def __init__(self, payoff_matrix, horizon=100, state_history=3):
        self.payoff_matrix = payoff_matrix
        self.horizon = horizon
        self.state_history = state_history
        self.state_size = state_history * 4  # 4 possible action pairs

    def reset(self):
        self.round = 0
        self.history = []
        return self._get_state()

    def _get_state(self):
        """State: one-hot encoding of last action pairs"""
        state = np.zeros(self.state_size, dtype=np.float32)

        for i in range(min(self.state_history, len(self.history))):
            action1, action2 = self.history[-(i+1)]
            idx = i * 4
            if action1 == 0 and action2 == 0:
                state[idx] = 1
            elif action1 == 0 and action2 == 1:
                state[idx + 1] = 1
            elif action1 == 1 and action2 == 0:
                state[idx + 2] = 1
            else:  # 1,1
                state[idx + 3] = 1

        return state

    def step(self, action1, action2):
        reward1 = float(self.payoff_matrix[action1, action2, 0])
        reward2 = float(self.payoff_matrix[action1, action2, 1])

        self.history.append((action1, action2))
        self.round += 1

        done = self.round >= self.horizon
        return self._get_state(), reward1, reward2, done, {}

