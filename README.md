# RL Nash with AIs (EUT) and Humans (CPT)

## Running the experiments:

### Run static games with python3 main_static.py

### Run repeated games with python3 main_repeated.py

#### Expect a series of prompts to customize the set up of the experiment, e.g. num episodes, game selection, agent selection, etc. 

## File Summaries:

### repeated_games/

#### Agents

AI: **ai_agent.py** implements epsilon greedy EUT learning, i.e. vanilla Q-learning

Learning Human (LH): **learning_human.py** implements epsilon greedy Q(s, a_i, a_-i) q learning, where each joint action pair is estimated. Beliefs about opponent actions are tracked with EMA. The update bootstraps with q(s', a') by integrating out the opponent actions using the beliefs, and then taking the value of the max action. Formally: q(s, a_i, a_-i) += alpha(reward + $\gamma$ * max(E_b{q(s', a'_i, a'\_-i)}) - q(s, a_i, a_-i)). Reference points are either fixed, or updated via EMA, max Q value, or opponent EMA (EMA over opponent received rewards). 

Aware Human (AH): **aware_human.py** implements a best reply agent *with no reinforcement learning*, although notably it does still update its reference poins the same way that LH does. The best reply algorithm first selects the opp best reply to each of the AH actions, then maximizes its action value over the opp BR to generate a response. 

All agents use a softmax tie break for when action probabilities are within $\tau$ of each other to randomize when there is no decisive action. 

#### Environment and Training

Game Env: **game_env.py** is where the 2x2 matrix games are stored. States are labeled with a base 4 encoding of joint actions, added together to turn a full history into a unique integer. Specifically, (0,0)->0 (0,1)->1 (1,0)->2 (1,1)->3. These joint action pair encodings are added into a state integer variable that is weighted by the place of the joint action in the history. 
          **double_action.py** uses a similar base n^2 state encoding system, where n is the action size for each player. The double auction game uses midpoint pricing if a trade goes through, and rewards of 0 otherwise (could do -1 to encourage exploration?)

Training: **
