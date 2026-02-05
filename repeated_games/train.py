import matplotlib.pyplot as plt
from .aware_human import AwareHumanPTAgent 
from .learning_human import LearningHumanPTAgent
from .ai_agent import AIAgent
from .game_env import RepeatedGameEnv 
from .utils import get_all_games
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
        'best_responses1': [],
        'best_responses2': [],
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

    agent1_type, agent2_type, pt_params, ref_setting, ref_lambda = br_dict['agent1_type'], br_dict['agent2_type'], br_dict['pt_params'], br_dict['ref_setting'], br_dict['ref_lambda']
 
    # For convergence tracking
    br1, br2 = best_responders(agent1, agent2, agent1_type, agent2_type, action_size, payoff_matrix, pt_params, env, ref_setting, ref_lambda)

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

            # Make sure best reply convergence tracking is conditioned on ref point changes
            br1.ref_update(payoff=reward1, state=state, opp_payoff=reward2)
            br2.ref_update(payoff=reward2, state=state, opp_payoff=reward1)

            # Then we update the opp reference point in BR and AH 
            if not isinstance(agent2, AIAgent):
                br1.opp_ref = agent2.ref_point
                if isinstance(agent1, AwareHumanPTAgent):
                    agent1.opp_ref = agent2.ref_point

            if not isinstance(agent1, AIAgent):
                br2.opp_ref = agent1.ref_point
                if isinstance(agent2, AwareHumanPTAgent):
                    agent2.opp_ref = agent1.ref_point

            global_step += 1
 
            # Store results
            episode_rewards1 += reward1
            episode_rewards2 += reward2

            episode_actions1.append(action1)
            episode_actions2.append(action2)

            episode_br1.append(br1.act())
            episode_br2.append(br2.act())
           

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

        results['best_responses1'].append(episode_br1)
        results['best_responses2'].append(episode_br2)
        
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

def analyze_matchup(results, agent1, agent2, agent1_type, agent2_type, game_name, games_dict, payoff_matrix, pt_params):
    """Comprehensive analysis of a matchup"""
    if game_name != 'Double Auction Game':
        actions = games_dict[game_name]['actions']
    else:
        actions = None

    fig = plt.figure(figsize=(15, 10))

    # 1. Learning curves
    ax1 = plt.subplot(3, 3, 1)
    window = 20
    smoothed1 = np.convolve(results['avg_rewards1'], np.ones(window)/window, mode='valid')
    smoothed2 = np.convolve(results['avg_rewards2'], np.ones(window)/window, mode='valid')

    ax1.plot(smoothed1, label=f'{agent1_type}', linewidth=2)
    ax1.plot(smoothed2, label=f'{agent2_type}', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward (smoothed)')
    ax1.set_title(f'Learning Curves\n{agent1_type} vs {agent2_type}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final strategy patterns
    ax2 = plt.subplot(3, 3, 2)

    if game_name != 'Double Auction Game':
        k = 100
    else:
        k = 1

    if len(results['ref_points1']) > 0:
        ref_points1 = results['ref_points1']
        ax2.plot(ref_points1[::k], label=f'{agent1_type}')

    if len(results['ref_points2']) > 0:
        ref_points2 = results['ref_points2']
        ax2.plot(ref_points2[::k], label=f'{agent2_type}')

    ax2.set_xlabel(f'Step (Every 100)')
    ax2.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x*k)}")
    )
    ax2.set_ylabel('Reference Point')
    ax2.set_title('Ref. Points Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Action distribution
    ax3 = plt.subplot(3, 3, 3)
    joint_actions = results['joint_actions']
    im = ax3.imshow(joint_actions, aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax3, label='Count')

    ax3.set_xticks(np.arange(joint_actions.shape[1]))
    ax3.set_xticklabels(np.arange(1, joint_actions.shape[1] + 1))

    ax3.set_yticks(np.arange(joint_actions.shape[0]))
    ax3.set_yticklabels(np.arange(1, joint_actions.shape[0] + 1))

    if game_name == 'Double Auction Game':
        ax3.set_xlabel(f'Seller ({agent2_type})')
        ax3.set_ylabel(f'Buyer ({agent1_type})')

    else:
        ax3.set_xlabel(f'{agent2_type}')
        ax3.set_ylabel(f'{agent1_type})')

    ax3.set_title("Joint Action Heatmap")

    # 4. Strategy evolution (for aware PT)
    ax4 = plt.subplot(3, 3, 4)
    if len(results['q_values1']) > 0:
        q_values1 = np.stack(results['q_values1'])
        print(f'unique q values agent 1: {np.unique(q_values1).size}, q vals 1 shape: {q_values1.shape}')
        error1 = np.mean(np.abs(q_values1 - payoff_matrix[:, :, agent1.agent_id]), axis=(1,2)) 
        if game_name == 'Double Auction Game':
            print(f"Error 1 shape: {error1.shape}, error 1 num unique: {np.unique(error1).size}, error 1 max: {error1.max()}, error 1 min: {error1.min()}")
        ax4.plot(error1, label=f'{agent1_type}', linewidth=1)
        
    else:
        q_values1 = []
        error1 = []

    if len(results['q_values2']) > 0:
        q_values2 = np.stack(results['q_values2'])
        error2 = np.mean(np.abs(q_values2 - payoff_matrix[:, :, agent2.agent_id]), axis=(1,2)) 
        if game_name == 'Double Auction Game':
            print(f"Error 2 shape: {error2.shape}, error 2 num unique: {np.unique(error2).size}, error 2 max: {error2.max()}, error 2 min: {error2.min()}")
        ax4.plot(error2, label=f'{agent2_type}', linewidth=1)
    else:
        q_values2 = []
        error2 = []

    ax4.set_xlabel('Step')
    ax4.set_ylabel('Mean Absolute Error (Q - R)')
    ax4.set_title('Mean Absolute Error')
    ax4.legend()
    
    # 5. Cumulative rewards
    ax5 = plt.subplot(3, 3, 5)
    cumulative1 = np.cumsum(results['rewards1'])
    cumulative2 = np.cumsum(results['rewards2'])

    ax5.plot(cumulative1, label=f'{agent1_type}', linewidth=2)
    ax5.plot(cumulative2, label=f'{agent2_type}', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Cumulative Reward')
    ax5.set_title('Cumulative Rewards')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Reward distribution
    ax6 = plt.subplot(3, 3, 6)
    if game_name != 'Double Auction Game':
        k = 100
    else:
        k = 1

    if len(results['q_values1']) > 1:
        q_values1 = np.stack(results['q_values1'])
        q_values1, q_values1_copy = q_values1[1:], q_values1[:-1]        
        q_values1_diff = q_values1 - q_values1_copy
        q_change1 = np.mean(np.abs(q_values1_diff), axis = tuple(range(1, q_values1_diff.ndim)))
        ax6.plot(q_change1[::k], label=f'{agent1_type}')
        print(q_change1.shape)

    if len(results['q_values2']) > 1:
        q_values2 = np.stack(results['q_values2'])
        q_values2, q_values2_copy = q_values2[1:], q_values2[:-1]     
        q_values2_diff = q_values2 - q_values2_copy
        q_change2 = np.mean(np.abs(q_values2_diff), axis = tuple(range(1, q_values2_diff.ndim)))
        ax6.plot(q_change2[::k], label=f'{agent2_type}')

    ax6.set_xlabel("Steps (every 100)")    
    ax6.set_ylabel('Q Value Diff')
    ax6.set_title('Q Values Changes')
    ax6.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x*k)}")
    )
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 3, 7)
    br1 = np.array(results['best_responses1'])
    br2 = np.array(results['best_responses2'])

    agent1_actions = np.array(results['actions1'])
    agent2_actions = np.array(results['actions2'])

    action_size = payoff_matrix.shape[0]

    assert len(br1) == len(br2) == len(agent1_actions) == len(agent2_actions), f"Lengths of lists don't match."
    T0 = len(br1[0])
    assert all(len(ep) == T0 for ep in br1), "br1 episodes have different lengths"
    assert all(len(ep) == T0 for ep in br2), "br2 episodes have different lengths"
    assert all(len(ep) == T0 for ep in agent1_actions), "agent1 episodes have different lengths"
    assert all(len(ep) == T0 for ep in agent2_actions), "agent2 episodes have different lengths"

    diff1 = []
    diff2 = []

    # Calculate similarity = 1 - TotalVariationDistance(a_i, br_i) 
    for episode in range(len(br1)):
        assert len(agent1_actions[episode]) == len(br1[episode]) == len(agent2_actions[episode]) == len(br2[episode]), "Episode list lengths don't match"
        br1_pi = np.zeros(action_size)
        br2_pi = np.zeros(action_size)
        a1_pi = np.zeros(action_size)
        a2_pi = np.zeros(action_size)

        for action in range(action_size):
            br1_pi[action] = sum(br1[episode] == action) / T0
            br2_pi[action] = sum(br2[episode] == action) / T0
            a1_pi[action] = sum(agent1_actions[episode] == action) / T0
            a2_pi[action] = sum(agent2_actions[episode] == action) / T0

        total_variation1 = 0
        total_variation2 = 0

        for action in range(action_size):
            total_variation1 += abs(br1_pi[action] - a1_pi[action])
            total_variation2 += abs(br2_pi[action] - a2_pi[action])

        similarity1 = 1 - 0.5 * total_variation1
        similarity2 = 1 - 0.5 * total_variation2

        diff1.append(similarity1)
        diff2.append(similarity2)

    ax7.plot(diff1, label=f'{agent1_type}')
    ax7.plot(diff2, label=f'{agent2_type}')
 
    ax7.set_xlabel("Episodes")
    ax7.set_ylabel("% BR actions")

    ax7.set_title('Convergence Between Agent Actions and BR')
    ax7.legend()
    ax7.grid(True, alpha=0.3)


    plt.suptitle(f'{game_name}: {agent1_type} vs {agent2_type}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "="*70)
    print(f"SUMMARY: {game_name} - {agent1_type} vs {agent2_type}")
    print("="*70)

    if results['avg_rewards1']:
        # Final performance
        final_episodes = 50
        if len(results['avg_rewards1']) >= final_episodes:
            final_avg1 = np.mean(results['avg_rewards1'][-final_episodes:])
            final_avg2 = np.mean(results['avg_rewards2'][-final_episodes:])
            std1 = np.std(results['avg_rewards1'][-final_episodes:])
            std2 = np.std(results['avg_rewards2'][-final_episodes:])

            print(f"\nFinal {final_episodes} episodes:")
            print(f"  {agent1_type}: {final_avg1:.3f} ± {std1:.3f}")
            print(f"  {agent2_type}: {final_avg2:.3f} ± {std2:.3f}")

            # Convergence check
            if std1 < 0.15 and std2 < 0.15:
                print("  ✓ Strategies converged (low variance)")
            else:
                print("  ⚠ Strategies still varying")

        # Action frequencies
        if results['actions1'] and game_name != 'Double Auction Game':
            last_10_actions1 = [a for ep in results['actions1'][-10:] for a in ep]
            last_10_actions2 = [a for ep in results['actions2'][-10:] for a in ep]

            freq1_0 = np.mean([a == 0 for a in last_10_actions1])
            freq2_0 = np.mean([a == 0 for a in last_10_actions2])

            print(f"\nFinal action frequencies (Action 0 = {actions[0]}):")
            print(f"  {agent1_type}: {freq1_0:.1%}")
            print(f"  {agent2_type}: {freq2_0:.1%}")

            # PT stats for learning PT agent
            if hasattr(agent1, 'get_pt_stats'):
                stats = agent1.get_pt_stats()
                print(f"\nPT Transformation Stats for {agent1_type}:")
                print(f"  Mean raw reward: {stats['mean_raw']:.3f}")
                print(f"  Mean PT reward: {stats['mean_pt']:.3f}")
                print(f"  PT amplification: {stats['mean_pt']/max(0.001, stats['mean_raw']):.2f}x")

            if hasattr(agent2, 'get_pt_stats'):
                stats = agent2.get_pt_stats()
                print(f"\nPT Transformation Stats for {agent2_type}:")
                print(f"  Mean raw reward: {stats['mean_raw']:.3f}")
                print(f"  Mean PT reward: {stats['mean_pt']:.3f}")
                print(f"  PT amplification: {stats['mean_pt']/max(0.001, stats['mean_raw']):.2f}x")

def run_complete_experiment(game_name, payoff_matrix, episodes=300, ref_setting='Fixed', pt_params={}):
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
            if agent2_type != "AI":
                opp_params['opp_ref'] = agent2.ref_point
            agent1 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=0, opp_params=opp_params,ref_setting=ref_setting, lambda_ref=ref_lambda)

        if agent2_type == 'Aware_PT':
            opp_params = dict()
            opp_params['opponent_type'] = agent1_type
            opp_params['opponent_action_size'] = action_size
            opp_params['opp_ref'] = None
            if agent1_type != "AI":
                opp_params['opp_ref'] = agent1.ref_point
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

    
def best_responders(agent1, agent2, agent1_type, agent2_type, action_size, payoff_matrix, pt_params, env, ref_setting, ref_lambda):
    opp_params = dict()
    opp_params['opponent_type'] = agent2_type
    opp_params['opponent_action_size'] = action_size
    opp_params['opp_ref'] = None
    if agent2_type != "AI":
        opp_params['opp_ref'] = agent2.ref_point
    best_responder1 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=0, opp_params=opp_params,ref_setting=ref_setting, lambda_ref=ref_lambda)

    opp_params = dict()
    opp_params['opponent_type'] = agent1_type
    opp_params['opponent_action_size'] = action_size
    opp_params['opp_ref'] = None
    if agent1_type != "AI":
        opp_params['opp_ref'] = agent1.ref_point
    best_responder2 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=1, opp_params=opp_params, ref_setting=ref_setting,                      lambda_ref=ref_lambda)

    return best_responder1, best_responder2
