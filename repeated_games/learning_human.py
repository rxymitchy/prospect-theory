from .ProspectTheory import ProspectTheory
import numpy as np
import random
from scipy.special import softmax

class LearningHumanPTAgent:
    """
    Learning Human PT Agent
    Doesn't know game structure, learns via RL
    Transforms rewards through PT
    """

    def __init__(self, state_size, action_size, opp_action_size, pt_params, agent_id=0, ref_setting='Fixed', lambda_ref=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.opp_action_size = opp_action_size
        self.pt = ProspectTheory(**pt_params)
        print('LH PT PARAMS: ', pt_params)
        self.agent_id = agent_id
        self.ref_point = pt_params['r']

        # Initialize beliefs function and q values as dictionaries
        self.beliefs = dict()
        self.q_values = dict()

        # Initialize belief and reference point lambda parameters (subject to tuning)
        self.lam_b = 0.95
        self.lam_r = lambda_ref

        # Set reference point update mode:
        self.ref_update_mode = ref_setting # options: Fixed, EMA, Q
        print(self.ref_update_mode, ref_setting)
       
        # Add an entry for each state populated with uniform probabilities over opponent action set size
        # And initialize q values
        for state in range(self.state_size):
            # Belief function
            self.beliefs[state] = np.ones(self.opp_action_size) / self.opp_action_size
            
            # Q-values
            self.q_values[state] = np.zeros((self.action_size, self.opp_action_size)) 

        self.state_visit_counter = dict()
        self.softmax_counter = 0

        # Q-learning parameters
        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.01

        # Pathology Detection parameters
        self.tau = 0.1
        ## subject to change
        self.temperature = 1.3 

        # Track raw vs PT rewards
        self.raw_rewards = []
        self.pt_rewards = []

    def act(self, state):
        # Epsilon exploration (lines 16-17 in alg 1)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Pathology detection (lines 14, 18, 19 in alg 1)
        action_values = self.calculate_action_values(state)

        ## Identify Optimal action
        optimal_action = np.argmax(action_values) 
        
        ## Identify second best action
        non_optimal_actions = action_values.copy()
        non_optimal_actions[optimal_action] = -np.inf
        second_best_action = non_optimal_actions.max()

        gap = action_values[optimal_action] - second_best_action
        #print(f"[Debug] gap value LH: {gap}")

        ## Check for pathology:
        if gap < self.tau:
            self.softmax_counter += 1
            ##Softmax
            ## Normalize 
            vals = action_values - action_values.max()
            action_probs = softmax(vals/self.temperature, axis=0)
            # Sample action
            action = np.random.choice(len(action_probs), p=action_probs)
            return action
        
        # Optimal Action (lins 20, 21 in alg 1)
        else:
            return int(optimal_action)

    def calculate_action_values(self, state):
        # Define action space
        action_values = np.zeros(self.action_size)
        ## Calculate V for each state, opp action tuple by forming a lottery 
        # with Q vals and beliefs
        for action in range(self.action_size):
            probabilities = self.beliefs[state] 
            assert np.isclose(probabilities.sum(), 1.0, atol=1e-5), \
            "Beliefs don't sum to 1"

            outcomes = self.q_values[state][action]

            action_val = self.pt.expected_pt_value(outcomes, probabilities) 
            action_values[action] = action_val

        return action_values

    def belief_update(self, state, opp_action):
        # Simple EMA for belief updates
        one_hot = np.zeros(self.opp_action_size)
        one_hot[opp_action] = 1
        self.beliefs[state] = self.lam_b * self.beliefs[state] + (1 - self.lam_b) * one_hot

    def ref_update(self, payoff, state, opp_payoff):
        # Just a debug check here
        if sum(self.state_visit_counter.values()) == 1:
            print(f"update mode: {self.ref_update_mode}")
        if self.ref_update_mode == "EMA":
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * payoff

        elif self.ref_update_mode == 'Q':
            # Set reference point to maximum, normalized q value
            weighted_q_vals = np.zeros(self.action_size)
            for action in range(self.action_size):
                # Beliefs are over opp actions, so we take the expectation for each action
                # over opp actions
                weighted_q_val = self.q_values[state][action] @ self.beliefs[state]
                weighted_q_vals[action] = weighted_q_val
            max_q_val = weighted_q_vals.max()
            
            # (1 - gamma) normalizes the future trajectory discounting that 
            # converges to 1/(1-gamma) when gamma < 1
            self.ref_point = (1 - self.gamma) * max_q_val

        elif self.ref_update_mode == 'EMAOR':
            # EMA, but now using the opponents rewards 
            # (to test to see if knowledge about how other player is doing changes behavior)
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * opp_payoff
            
        # Make sure to update ref point in pt function
        self.pt.r = self.ref_point

    def q_value_update(self, state, next_state, action, opp_action, reward, done=False):
        '''The point here is to align our agent with PT-EB principles. We treat our own decision making
        as certain (so q values are in reward/outcome space), but we allot our uncertainty to opp actions.
        That is why you will see the expectation taken over opponent actions conditioned on our beliefs
        in the boot strap. 
        '''
        if state not in self.state_visit_counter.keys():
            self.state_visit_counter[state] = 0

        self.state_visit_counter[state] += 1

        ## next state is necessary for double auction game
        q_values = self.q_values[next_state]
        beliefs = self.beliefs[next_state]
        #print(f"beliefs: {beliefs}")

        # Get maximuj value (not index)
        ## - inf because rewards can be negative
        optimal_next_q_value = -np.inf

        eps = 1e-8 # For tie breaks
  
        # Integrate out opp actions (Q(s, a_i, a_-i) -> Q(s, a_i)) in line with PT EB philosophy
        # Then get the max for the bellman update (max Q(s, a))
        for a_prime in range(self.action_size):
            q_val = q_values[a_prime]

            # linear expectation of beliefs and values (integrate out opp acts)
            weighted_q_val = np.dot(beliefs, q_val)

            # We are maximizing for bellman update
            if weighted_q_val > optimal_next_q_value + eps:
                optimal_next_q_value = weighted_q_val

            # Tie breaker (randomly choose, maybe random is not the right choice here? Should average out)
            elif np.abs(weighted_q_val - optimal_next_q_value) <= eps:
                if random.random() < 0.5:
                    optimal_next_q_value = weighted_q_val

        # Get stored value (state, joint action value) for bootstrap 
        q_value = self.q_values[state][action][opp_action]

        if done:
            optimal_next_q_value = 0.0

        # Calculate delta in untransformed reward space
        delta = reward + self.gamma * optimal_next_q_value - q_value 
        # Update q values
        self.q_values[state][action][opp_action] += self.alpha * delta

    # Deprecated from the q value convergence metric i was talking about
    def get_q_values(self):
        q_values = np.zeros((self.action_size, self.opp_action_size))

        total_visits = sum(self.state_visit_counter.values())

        if total_visits == 0:
            return q_values

        for state, q_vals in self.q_values.items():
            num_visits = self.state_visit_counter.get(state, 0)

            if num_visits == 0:
                continue

            weight = num_visits / total_visits

            q_values += weight * q_vals

        return q_values




