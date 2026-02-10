import matplotlib.pyplot as plt
from .aware_human import AwareHumanPTAgent 
from .learning_human import LearningHumanPTAgent
from .ai_agent import AIAgent
from .game_env import RepeatedGameEnv 
from .utils import get_all_games, best_responders
from .analyze import analyze_matchup
import numpy as np
import pandas as pd
import time

from matplotlib.ticker import FuncFormatter

def train_agents(agent1, agent2, env, br_dict, episodes=500,
                 exploration_decay=0.995, verbose=True, game_name=''):
    """
    Train two agents against each other
    """

    results = {
        'rewards1': [],
        'rewards2': [],
        'actions1': [],
        'actions2': [],
        'avg_rewards1': [],
        'avg_rewards2': [],
        'strategies1': [],  # For aware PT agent
        'strategies2': [],
        'q_values1': [],
        'q_values2':[],
        'ref_points1': [],
        'ref_points2': [],
    }
    joint_counts = np.zeros((agent1.action_size,agent2.action_size), dtype=int)

    start_time = time.time()
    last_time = 0

    log_every = 100
    global_step = 0

    # Defined to initialize BR agents, agent1 and agent 2 actions sizes always the same for this setup
    action_size = agent1.action_size
    if game_name == 'Double Auction Game':
        payoff_matrix = env.build_payoff_matrix()

    else:
        payoff_matrix = env.payoff_matrix


    for episode in range(episodes):
        state = env.reset()
        episode_rewards1 = 0
        episode_rewards2 = 0
        episode_actions1 = []
        episode_actions2 = []
        episode_q_values1 = []
        episode_q_values2 = []
        episode_br1 = []
        episode_br2 = []

        for _ in range(env.horizon):
            # Agent 1 chooses action
            action1 = agent1.act(state)

            # Agent 2 chooses action
            action2 = agent2.act(state)

            # Execute step
            next_state, reward1, reward2, done, _ = env.step(action1, action2)

            if isinstance(agent1, LearningHumanPTAgent):
                # Updates
                agent1.belief_update(state, action2)
                agent1.ref_update(payoff=reward1, state=state, opp_payoff=reward2)
                agent1.q_value_update(state, next_state, action1, action2, reward1, done)

                # Get q vals, normalize by multiplying by 1 - gamma to remove future discounting
                q_vals = agent1.get_q_values()
                q_vals = np.asarray(q_vals, dtype=np.float32)  
                q_vals = (1 - agent1.gamma) * q_vals

                # Track values of interest

                # Tracking
                if game_name == 'Double Auction Game':
                    if global_step % log_every == 0:
                        results['q_values1'].append(q_vals)
                        results['ref_points1'].append(agent1.ref_point)
                else:
                    results['q_values1'].append(q_vals)
                    results['ref_points1'].append(agent1.ref_point)

                del q_vals

            elif isinstance(agent1, AIAgent):
                # Update code here
                agent1.update(state, action1, next_state, reward1, done)

                # Get q vals, normalize by multiplying by 1 - gamma to remove future discounting
                q_vals = agent1.get_q_values()
                q_vals = np.asarray(q_vals, dtype=np.float32)
                q_vals = (1 - agent1.gamma) * q_vals

                # Tracking
                if game_name == 'Double Auction Game':
                    if global_step % log_every == 0:
                        results['q_values1'].append(q_vals)
                else:
                    results['q_values1'].append(q_vals)
                
                del q_vals

            else: # Aware Human
                agent1.ref_update(payoff=reward1, state=state, opp_payoff=reward2)
                results['ref_points1'].append(agent1.ref_point)
                

            if isinstance(agent2, LearningHumanPTAgent):
                # Update
                agent2.belief_update(state, action1)
                agent2.ref_update(payoff=reward2, state=state, opp_payoff=reward1)
                agent2.q_value_update(state, next_state, action2, action1, reward2, done)

                # Get q vals, normalize by multiplying by 1 - gamma to remove future discounting
                q_vals = agent2.get_q_values()
                q_vals = np.asarray(q_vals, dtype=np.float32)
                q_vals = (1 - agent2.gamma) * q_vals

                # Tracking
                if game_name == 'Double Auction Game':
                    if global_step % log_every == 0:
                        results['q_values2'].append(q_vals)
                        results['ref_points2'].append(agent2.ref_point)
                else:
                    results['q_values2'].append(q_vals)
                    results['ref_points2'].append(agent2.ref_point)
                del q_vals

            elif isinstance(agent2, AIAgent):
                # Update code here
                agent2.update(state, action2, next_state, reward2, done)
                q_vals = agent2.get_q_values()
                q_vals = np.asarray(q_vals, dtype=np.float32)
                q_vals = (1 - agent2.gamma) * q_vals

                # Tracking
                if game_name == 'Double Auction Game':
                    if global_step % log_every == 0:
                        results['q_values2'].append(q_vals)
                else:
                    results['q_values2'].append(q_vals)

                del q_vals

            else: # Aware Human
                agent2.ref_update(payoff=reward2, state=state, opp_payoff=reward1)
                results['ref_points2'].append(agent2.ref_point)


            global_step += 1
 
            # Store results
            episode_rewards1 += reward1
            episode_rewards2 += reward2

            episode_actions1.append(action1)
            episode_actions2.append(action2)

            joint_counts[action1, action2] += 1

            state = next_state

            if done:
                break

        # Store episode results
        print("out of loop")
        steps = env.horizon
        avg_reward1 = episode_rewards1 / steps
        avg_reward2 = episode_rewards2 / steps

        results['rewards1'].append(episode_rewards1)
        results['rewards2'].append(episode_rewards2)

        results['actions1'].append(episode_actions1)
        results['actions2'].append(episode_actions2)

        results['avg_rewards1'].append(avg_reward1)
        results['avg_rewards2'].append(avg_reward2)
        
        # Decay exploration
        if isinstance(agent1, (LearningHumanPTAgent, AIAgent)):
            agent1.epsilon = max(agent1.epsilon * exploration_decay, agent1.epsilon_min)

        if isinstance(agent2, (LearningHumanPTAgent, AIAgent)):
            agent2.epsilon = max(agent2.epsilon * exploration_decay, agent2.epsilon_min)

        # Progress update
        if verbose and (episode + 1) % 100 == 0:
            print(f"  Episode {episode + 1}/{episodes}: "
                  f"Avg rewards = {avg_reward1:.3f}, {avg_reward2:.3f}"
                  f"\n Time since start = {time.time() - start_time}, Time this 100 episodes = {time.time() - last_time}")

        last_time = time.time()

    print("joint actions: ", joint_counts)
    results['joint_actions'] = joint_counts

    return results

def run_complete_experiment(game_name, payoff_matrix, episodes=300, ref_setting='Fixed', pt_params={}, ref_point=0):
    """Run all agent matchups for a game"""

    print("\n" + "="*80)
    print(f"COMPLETE EXPERIMENT: {game_name}")
    print("="*80)

    # Reference point setting
    # Options = Fixed, EMA, Q, 'EMAOR': EMA of Opp rewards
    ref_lambda = 0.9

    state_history_len = 2

    # Define all matchups to test
    matchups = [
        ('Aware_PT', 'AI'),
        ('LH', 'AI'),
        ('Aware_PT', 'LH'),
        ('Aware_PT', 'Aware_PT'), 
        ('LH', 'LH'),
        ('AI', 'AI')  # Baseline
    ]

    all_results = {}

    for agent1_type, agent2_type in matchups:
        print(f"\n{'='*70}")
        print(f"MATCHUP: {agent1_type} vs {agent2_type}")
        print('='*70)

        # Reset environment
        env = RepeatedGameEnv(payoff_matrix, horizon=100, state_history=state_history_len)

        # Create agents based on type
        ## 2x2 games only
        action_size = 2
        if agent1_type == 'LH':
            agent1 = LearningHumanPTAgent(env.state_size, action_size, action_size, pt_params, agent_id=0, ref_setting=ref_setting, lambda_ref = ref_lambda)
        elif agent1_type == 'AI':  # AI
            agent1 = AIAgent(env.state_size, action_size, action_size, agent_id=0)

        if agent2_type == 'LH':
            agent2 = LearningHumanPTAgent(env.state_size, action_size, action_size, pt_params, agent_id=1, ref_setting=ref_setting, lambda_ref=ref_lambda)
        elif agent2_type == 'AI':  # AI
            agent2 = AIAgent(env.state_size, action_size, action_size, agent_id=1)

        if agent1_type == 'Aware_PT':
            opp_params = dict()
            opp_params['opponent_type'] = agent2_type
            opp_params['opponent_action_size'] = action_size
            opp_params['opp_ref'] = None

            if agent2_type != "AI": # PT agent
                if agent2_type == "Aware_PT":
                    opp_params['opp_ref'] = ref_point 
                    opp_params['opp_pt'] = pt_params
                else:
                    opp_params['opp_ref'] = agent2.ref_point

            agent1 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=0, opp_params=opp_params,ref_setting=ref_setting, lambda_ref=ref_lambda)

        if agent2_type == 'Aware_PT':
            opp_params = dict()
            opp_params['opponent_type'] = agent1_type
            opp_params['opponent_action_size'] = action_size
            opp_params['opp_ref'] = None

            if agent2_type != "AI": # PT agent
                if agent2_type == "Aware_PT":
                    opp_params['opp_ref'] = ref_point
                    opp_params['opp_pt'] = pt_params
                else:
                    opp_params['opp_ref'] = agent2.ref_point

            agent2 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=1, opp_params=opp_params, ref_setting=ref_setting,                      lambda_ref=ref_lambda)


        # Train the matchup
        print(f"Training {episodes} episodes...")
        results = train_agents(agent1, agent2, agent1_type, agent2_type, env, episodes=episodes, verbose=True, game_name=game_name)

        # Store results
        matchup_key = f"{agent1_type}_vs_{agent2_type}"
        all_results[matchup_key] = {
            'results': results,
            'agent1': agent1,
            'agent2': agent2
        }

        # Analyze this matchup
        games_dict = get_all_games()
        analyze_matchup(results, agent1, agent2, agent1_type, agent2_type, game_name, games_dict, payoff_matrix, pt_params)

    return all_results

def compare_all_results(all_results, game_name):
    """Compare performance across all matchups"""

    print("\n" + "="*80)
    print(f"COMPARISON ACROSS ALL MATCHUPS: {game_name}")
    print("="*80)

    comparison_data = []

    for matchup_key, data in all_results.items():
        results = data['results']

        if results['avg_rewards1'] and len(results['avg_rewards1']) >= 50:
            final_avg1 = np.mean(results['avg_rewards1'][-50:])
            final_avg2 = np.mean(results['avg_rewards2'][-50:])
            std1 = np.std(results['avg_rewards1'][-50:])
            std2 = np.std(results['avg_rewards2'][-50:])

            comparison_data.append({
                'Matchup': matchup_key,
                'Agent1_Avg': final_avg1,
                'Agent1_Std': std1,
                'Agent2_Avg': final_avg2,
                'Agent2_Std': std2,
                'Total_Avg': (final_avg1 + final_avg2) / 2,
                'Difference': abs(final_avg1 - final_avg2)
            })

    # Create comparison table
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Total_Avg', ascending=False)

        print("\nPerformance Comparison (last 50 episodes):")
        print(df.to_string(index=False))

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot of average rewards
        matchups = df['Matchup'].tolist()
        agent1_avgs = df['Agent1_Avg'].tolist()
        agent2_avgs = df['Agent2_Avg'].tolist()

        x = np.arange(len(matchups))
        width = 0.35

        ax1.bar(x - width/2, agent1_avgs, width, label='Agent 1', alpha=0.7)
        ax1.bar(x + width/2, agent2_avgs, width, label='Agent 2', alpha=0.7)
        ax1.set_xlabel('Matchup')
        ax1.set_ylabel('Average Reward')
        ax1.set_title(f'Average Rewards by Matchup\n{game_name}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(matchups, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Heatmap of total average
        matchup_matrix = np.zeros((3, 3))
        matchup_names = ['Aware_PT', 'Learning_PT', 'AI']

        for i, agent1 in enumerate(matchup_names):
            for j, agent2 in enumerate(matchup_names):
                key = f"{agent1}_vs_{agent2}"
                if key in all_results:
                    results = all_results[key]['results']
                    if results['avg_rewards1']:
                        avg1 = np.mean(results['avg_rewards1'][-50:])
                        avg2 = np.mean(results['avg_rewards2'][-50:])
                        matchup_matrix[i, j] = (avg1 + avg2) / 2

        im = ax2.imshow(matchup_matrix, cmap='viridis')
        ax2.set_xticks(range(len(matchup_names)))
        ax2.set_yticks(range(len(matchup_names)))
        ax2.set_xticklabels(matchup_names)
        ax2.set_yticklabels(matchup_names)
        ax2.set_title('Total Average Reward Heatmap')

        # Add text annotations
        for i in range(len(matchup_names)):
            for j in range(len(matchup_names)):
                text = ax2.text(j, i, f'{matchup_matrix[i, j]:.2f}',
                              ha="center", va="center", color="w")

        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        plt.show()

    return comparison_data

    
