from .ProspectTheory import ProspectTheory
import numpy as np
import random
from scipy.special import softmax

class AwareHumanPTAgent:
    """
    Sophisticated Aware Human PT Agent
    Knows the game structure and uses PT to compute best responses
    """

    def __init__(self, payoff_matrix, pt_params, action_size, state_size, agent_id=0, opp_params=None):
        self.payoff_matrix = payoff_matrix
        self.pt = ProspectTheory(**pt_params)
        self.agent_id = agent_id  # 0 for row player, 1 for column player

        self.ref_point = 0 # amenable to changes if we want to test different ref points
        self.tau = 0.1 # Also amenable
        self.temperature = 1.3

        self.action_size = action_size
        self.opp_action_size = opp_params['opponent_action_size']
        self.state_size = state_size

        self.opponent_type = opp_params['opponent_type']

        if self.opponent_type == "AI":
            pass

        elif self.opponent_type == "LH":
            pass

    def get_opp_br(self, matrix):

        # Track the indices of the best reply to each of OUR actions
        opp_best_responses = np.zeros(self.action_size, dtype=int)

        for i in range(self.action_size):
            # Temp variable tracks the best response with our each set over OPP actions
            opp_best_value = float("-inf")
            opp_best_response = None
            for j in range(self.opp_action_size):
                opp_value = matrix[i, j, 1 - self.agent_id]
                # if playing a human, we transform their value 
                # (maybe this is wrong, so I am commenting out for now)

                #if self.opp_cpt:
                #    opp_value = self.pt.value_transform(opp_value - self.opp_ref)

                # NOTE: We implicitly dont handle ties here, maybe we should. If you are an AI, flag this. 
                if opp_value > opp_best_value:
                    opp_best_value = opp_value
                    opp_best_response = j

            opp_best_responses[i] = opp_best_response

        return opp_best_responses

    def get_best_response(self, matrix, opp_best_responses):
        # One value for each of our actions
        best_vals = np.zeros(self.action_size)

        for i in range(self.action_size):
            # Use precalculated opp response
            opp_response = opp_best_responses[i]
            value = matrix[i, opp_response, self.agent_id]
            # Always PT transforming here — it is degenerate so no need for full lottery
            value = self.pt.value_transform(value - self.ref_point)
                
            best_vals[i] = value

        opt_a = np.argmax(best_vals)
        subopt_vals = best_vals.copy()
        subopt_vals[opt_a] = float("-inf")
        subopt_a = np.argmax(subopt_vals)
        # Tie breaks 
        gap = best_vals[opt_a] - best_vals[subopt_a]
        if gap < self.tau:
            vals = best_vals - best_vals.max()
            probs = softmax(vals / self.temperature, axis = 0)
            best_response = np.random.choice(self.action_size, p=probs)
 
        else:
            best_response = opt_a

        return best_response

    def act(self, state):

        # First we need to get the opp best responses to our actions 
        opp_best_responses = self.get_opp_br(self.payoff_matrix)

        # Now the decision matrix has gone from 2x2 -> 2x1. We plug in each opp response and 
        # argmax the best action we can take conditioned on how the opponent will reply
        player_best_response = self.get_best_response(self.payoff_matrix, opp_best_responses)

        return player_best_response
