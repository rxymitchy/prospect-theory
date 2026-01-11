import numpy as np

class ProspectTheory:
    """Complete PT implementation for game analysis"""

    def __init__(self, lambd=2.25, alpha=0.88, gamma=0.61, r=0, delta=0.69):
        self.lambd = lambd  # Loss aversion
        self.alpha = alpha  # Diminishing sensitivity
        self.gamma = gamma  # Probability weighting
        self.delta = delta
        self.r = r          # Reference point

    def value_function(self, x):
        """PT value function v(x)"""
        if x >= self.r:
            return (x + 1e-10) ** self.alpha
        else:
            return -self.lambd * ((-x + 1e-10) ** self.alpha)

    def w_plus(self, p):
        """Prelec probability weighting function"""
        # Why are we using the Prelec function but defining a different one in the paper?
        if p <= 0:
            return 0
        elif p >= 1:
            return 1
        else:
            return p ** self.gamma / (p ** gamma + (1 - p) ** self.gamma) ** (1 / self.gamma) 

    def cpt_gains(outcomes, probabilities, w_plus, v):
        # keep gains only
        mask = outcomes > self.r
        x = outcomes[mask]
        p = probabilities[mask]

        if len(x) == 0:
            return 0.0

        # sort gains ascending
        order = np.argsort(x)
        x = x[order]
        p = p[order]

        # tail probabilities
        tail = np.cumsum(p[::-1])[::-1]

        # decision weights
        w_tail = w_plus(tail)
        w_tail_next = np.concatenate([w_tail[1:], [0.0]])
        pi = w_tail - w_tail_next

        return np.sum(pi * self.value_function(x))


