from .ProspectTheory import ProspectTheory
import numpy as np
import random
from scipy.special import softmax

class AwareHumanPTAgent:
    """
    Sophisticated Aware Human PT Agent
    Knows the game structure and uses PT to compute best responses
    """

    def __init__(self, payoff_matrix, pt_params, action_size, state_size, agent_id=0, opp_params=None, ref_setting='Fixed', lambda_ref=0.95):
        self.payoff_matrix = payoff_matrix
        self.pt = ProspectTheory(**pt_params)
        print('AH PT PARAMS: ', pt_params)

        self.agent_id = agent_id  # 0 for row player, 1 for column player
        self.lam_r = lambda_ref
        self.ref_update_mode = ref_setting
        print('AH: ', self.ref_update_mode)

        self.ref_point = 0 # amenable to changes if we want to test different ref points
        self.tau = 0.1 # Also amenable
        self.temperature = 1.3

        self.action_size = action_size
        self.opp_action_size = opp_params['opponent_action_size']
        self.state_size = state_size

        self.opponent_type = opp_params['opponent_type']
        self.opp_pt = opp_params['opp_pt']

        if self.opponent_type == "AI":
            self.opp_cpt = False

        else:
            self.opp_cpt = True
            self.opp_ref = opp_params["opp_ref"]

        self.softmax_counter = 0 

    def get_opp_br(self, matrix):

        # Track the indices of the best reply to each of OUR actions
        opp_best_responses = np.zeros(self.action_size, dtype=int)

        for i in range(self.action_size):
            # Temp variable tracks the best response with our each set over OPP actions
            opp_best_value = float("-inf")
            opp_best_response = None
            for j in range(self.opp_action_size):
                opp_value = matrix[i, j, 1 - self.agent_id]

                if self.opp_cpt:
                    opp_value = self.pt.value_function(opp_value - self.opp_ref)

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
            value = self.pt.value_function(value - self.ref_point)
            best_vals[i] = value

        

        opt_a = np.argmax(best_vals)
        subopt_vals = best_vals.copy()
        subopt_vals[opt_a] = float("-inf")
        subopt_a = np.argmax(subopt_vals)
        # Tie breaks 
        gap = best_vals[opt_a] - best_vals[subopt_a]
        if gap < self.tau:
            # Log tie break
            self.softmax_counter += 1

            vals = best_vals - best_vals.max()
            probs = softmax(vals / self.temperature, axis = 0)
            best_response = np.random.choice(self.action_size, p=probs)
 
        else:
            best_response = opt_a

        return best_response

    def act(self, state=None):
        matrix = self.payoff_matrix
        if self.agent_id == 1:
            matrix = matrix.transpose(1, 0, 2)


        # First we need to get the opp best responses to our actions 
        opp_best_responses = self.get_opp_br(matrix)

        # Now the decision matrix has gone from 2x2 -> 2x1. We plug in each opp response and 
        # argmax the best action we can take conditioned on how the opponent will reply
        player_best_response = self.get_best_response(matrix, opp_best_responses)

        return player_best_response

    def ref_update(self, payoff, state, opp_payoff):
        if self.ref_update_mode == "EMA":
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * payoff

        elif self.ref_update_mode == 'Q':
            self.ref_point = self.payoff_matrix.max()

        elif self.ref_update_mode == 'EMAOR':
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * opp_payoff
