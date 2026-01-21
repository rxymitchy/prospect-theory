import numpy as np # type: ignore
from .ProspectTheory import ProspectTheory


class FictitiousPlayAgent:
    """
    Classical Fictitious Play agent
    Tracks empirical frequency of opponent actions
    Best-responds using either Expected Utility or Prospect Theory
    """
    
    def __init__(self, payoff_matrix, agent_id=0, use_pt=False, pt_params=None):
        """
        Initialize Fictitious Play agent
        
        Parameters:
        -----------
        payoff_matrix : ndarray (n_actions1, n_actions2, 2)
            Game payoff matrix
        agent_id : int
            0 for row player, 1 for column player
        use_pt : bool
            Whether to use PT for evaluations (True) or EU (False)
        pt_params : dict
            PT parameters if use_pt=True
        """
        self.payoff_matrix = payoff_matrix
        self.agent_id = agent_id
        self.use_pt = use_pt
        
        if use_pt:
            self.pt = ProspectTheory(**pt_params)
        
        # Game dimensions
        if agent_id == 0:  # Row player
            self.n_actions = payoff_matrix.shape[0]
            self.opp_n_actions = payoff_matrix.shape[1]
        else:  # Column player
            self.n_actions = payoff_matrix.shape[1]
            self.opp_n_actions = payoff_matrix.shape[0]
        
        # Track counts of opponent actions
        self.opponent_counts = np.zeros(self.opp_n_actions)
        self.round = 0
        
        # Payoff matrix for this agent
        if agent_id == 0:
            self.my_payoffs = payoff_matrix[:, :, 0]  # Row player payoffs
        else:
            self.my_payoffs = payoff_matrix[:, :, 1]  # Column player payoffs
    
    def act(self, state=None):
        """
        Choose action based on empirical opponent distribution
        
        Parameters:
        -----------
        state : int (ignored in FP, kept for interface compatibility)
        
        Returns:
        --------
        int : chosen action
        """
        if self.round == 0:
            # First round: random action
            return np.random.randint(self.n_actions)
        
        # Empirical distribution of opponent actions
        empirical_dist = self.opponent_counts / self.round
        
        if self.use_pt:
            # PT-based best response
            return self._pt_best_response(empirical_dist)
        else:
            # EU-based best response
            return self._eu_best_response(empirical_dist)
    
    def _eu_best_response(self, opp_dist):
        """Expected Utility best response"""
        if self.agent_id == 0:
            # Row player: my_payoffs is n_actions x opp_n_actions
            expected_payoffs = self.my_payoffs @ opp_dist
        else:
            # Column player: my_payoffs is opp_n_actions x n_actions
            expected_payoffs = self.my_payoffs.T @ opp_dist
        
        return np.argmax(expected_payoffs)
    
    def _pt_best_response(self, opp_dist):
        """Prospect Theory best response"""
        best_action = 0
        best_value = -float('inf')
        
        for my_action in range(self.n_actions):
            # Get payoffs for this action against opponent's distribution
            if self.agent_id == 0:
                outcomes = self.my_payoffs[my_action, :]
            else:
                outcomes = self.my_payoffs[:, my_action]
            
            # Calculate PT value
            pt_value = self.pt.expected_pt_value(outcomes, opp_dist)
            
            if pt_value > best_value:
                best_value = pt_value
                best_action = my_action
        
        return best_action
    
    def update(self, opp_action, state=None):
        """
        Update empirical counts with opponent's action
        
        Parameters:
        -----------
        opp_action : int
            Opponent's action in last round
        state : int (ignored, kept for interface)
        """
        self.opponent_counts[opp_action] += 1
        self.round += 1
    
    def reset(self):
        """Reset for new episode"""
        self.opponent_counts = np.zeros(self.opp_n_actions)
        self.round = 0
    
    def get_empirical_distribution(self):
        """Get current empirical distribution of opponent actions"""
        if self.round == 0:
            return np.ones(self.opp_n_actions) / self.opp_n_actions
        return self.opponent_counts / self.round


class SmoothFictitiousPlayAgent(FictitiousPlayAgent):
    """
    Smooth Fictitious Play with exploration
    Uses softmax instead of strict best response
    """
    
    def __init__(self, payoff_matrix, agent_id=0, use_pt=False, 
                 pt_params=None, temperature=0.1):
        super().__init__(payoff_matrix, agent_id, use_pt, pt_params)
        self.temperature = temperature
    
    def act(self, state=None):
        """Softmax action selection"""
        if self.round == 0:
            return np.random.randint(self.n_actions)
        
        empirical_dist = self.opponent_counts / self.round
        
        # Calculate action values
        action_values = np.zeros(self.n_actions)
        for my_action in range(self.n_actions):
            if self.agent_id == 0:
                outcomes = self.my_payoffs[my_action, :]
            else:
                outcomes = self.my_payoffs[:, my_action]
            
            if self.use_pt:
                action_values[my_action] = self.pt.expected_pt_value(outcomes, empirical_dist)
            else:
                action_values[my_action] = np.dot(outcomes, empirical_dist)
        
        # Softmax probabilities
        exp_values = np.exp(action_values / self.temperature)
        probs = exp_values / np.sum(exp_values)
        
        return np.random.choice(self.n_actions, p=probs)