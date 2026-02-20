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

#### Environment, Analysis, and Training

Game Env: **game_env.py** is where the 2x2 matrix games are stored. States are labeled with a base 4 encoding of joint actions, added together to turn a full history into a unique integer. Specifically, (0,0)->0 (0,1)->1 (1,0)->2 (1,1)->3. These joint action pair encodings are added into a state integer variable that is weighted by the place of the joint action in the history. 
          **double_action.py** uses a similar base n^2 state encoding system, where n is the action size for each player. The double auction game uses midpoint pricing if a trade goes through, and rewards of 0 otherwise (could do -1 to encourage exploration?)

Training: **train.py** train_agents() initializes the environment, tracks rewards, actions, q values and ref points for each agent, and steps through the environment episodically. Each round played updates reference points, beliefs, and q values. Importantly, the AH gets the opponent reference point updated each round for its calculus. Epsilon is updated at the end of each episode with a minimum of 0.01.
                       run_complete_experiment() is a wrapper for train_agents, automating the full cycle through the loop. I currently a not using this function, which is why some of the code is a little dubious (e.g. the matchups dont account for the agent typs being on the other side, that is row/col and also col/row). It's probably not worth checking this, I am going to stay in the custom matchup logic and manually recombine. 

Analysis: **analyze.py** analyze_matchup() includes a cumulative rewards/time plot, a ref point over time plot, an action heat map, a Q-value convergence plot (how are q values converging to the real payoffs), a learning convergence plot (how much are q values changing), and then action frequency charts for each player. 
                         compare_all_results() is essentially deprecated and a part of the run complete experiment wrapper. 
