class ProspectTheory:
    """Complete PT implementation for game analysis"""

    def __init__(self, lambd=2.25, alpha=0.88, gamma=0.61, r=0):
        self.lambd = lambd  # Loss aversion
        self.alpha = alpha  # Diminishing sensitivity
        self.gamma = gamma  # Probability weighting
        self.r = r          # Reference point

    def value_function(self, x):
        """PT value function v(x)"""
        if x >= 0:
            return (x + 1e-10) ** self.alpha
        else:
            return -self.lambd * ((-x + 1e-10) ** self.alpha)

    def probability_weighting(self, p):
        """Prelec probability weighting function"""
        # Why are we using the Prelec function but defining a different one in the paper?
        if p <= 0:
            return 0
        elif p >= 1:
            return 1
        else:
            return np.exp(-np.power(-np.log(p), self.gamma))

    def expected_pt_value(self, outcomes, probabilities):
        """Calculate PT expected value with probability weighting"""
        ''' We need to differentiate between gains and losses, taking the cdf for
        losses only and the decumulative df for gains. Right now this takes the cdf
        for everything.
        '''
        weighted_sum = 0
        sorted_indices = np.argsort(outcomes)[::-1]  # Sort descending

        for idx in sorted_indices:
            outcome = outcomes[idx]
            prob = probabilities[idx]
            if prob > 0:
                # Cumulative probability for rank-dependent weighting
                # We need to take the decumulative density function for gains
                rank = np.sum(probabilities[sorted_indices[:idx+1]])
                prev_rank = rank - prob

                # Decision weight = w(rank) - w(prev_rank)
                # We need to separate this into curr - prev and curr + next (M^- + M^+)
                weight = self.probability_weighting(rank) - self.probability_weighting(prev_rank)
                weighted_sum += weight * outcome

        return weighted_sum
