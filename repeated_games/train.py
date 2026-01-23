import matplotlib.pyplot as plt
from .aware_human import AwareHumanPTAgent 
from .learning_human import LearningHumanPTAgent
from .ai_agent import AIAgent
from .game_env import RepeatedGameEnv 
from .utils import get_all_games
import numpy as np
import pandas as pd


def train_agents(agent1, agent2, env, episodes=500,
                 exploration_decay=0.995, verbose=True):
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
        'strategies2': []
    }

    for episode in range(episodes):
        state = env.reset()
        episode_rewards1 = 0
        episode_rewards2 = 0
        episode_actions1 = []
        episode_actions2 = []
        episode_q_values1 = dict()
        episode_q_values2 = dict()

        for _ in range(env.horizon):
            # Agent 1 chooses action
            action1 = agent1.act(state)

            # Agent 2 chooses action
            action2 = agent2.act(state)

            # Execute step
            next_state, reward1, reward2, done, _ = env.step(action1, action2)

            if isinstance(agent1, LearningHumanPTAgent):
                agent1.belief_update(state, action2)
                agent1.ref_update(reward1)
                agent1.q_value_update(state, next_state, action1, action2, reward1)
                if state not in episode_q_values1.keys():
                    episode_q_values1[state] = []

            elif isinstance(agent1, AIAgent):
                # Update code here
                agent1.update(state, action1, next_state, reward1)

            if isinstance(agent2, LearningHumanPTAgent):
                agent2.belief_update(state, action1)
                agent2.ref_update(reward2)
                agent2.q_value_update(state, next_state, action2, action1, reward2)

            elif isinstance(agent2, AIAgent):
                # Update code here
                agent2.update(state, action2, next_state, reward2)

            # Keep Aware agents in the loop regarding updates
            if isinstance(agent1, AwareHumanPTAgent):
                if isinstance(agent2, LearningHumanPTAgent):
                    opp_params = {'type':'LH', 'beliefs':agent2.beliefs, 'q_values':agent2.q_values}
                    agent1.update(opp_params)
 
                elif isinstance(agent2, AIAgent):
                    opp_params = {'type':'AI', 'q_values':agent2.q_values}
                    agent1.update(opp_params)
           
            if isinstance(agent2, AwareHumanPTAgent):
                if isinstance(agent1, LearningHumanPTAgent):
                    opp_params = {'type':'LH', 'beliefs':agent1.beliefs, 'q_values':agent1.q_values}
                    agent2.update(opp_params)

                elif isinstance(agent1, AIAgent):
                    opp_params = {'type':'AI', 'q_values':agent1.q_values}
                    agent2.update(opp_params)

            # Store results
            episode_rewards1 += reward1
            episode_rewards2 += reward2
            episode_actions1.append(action1)
            episode_actions2.append(action2)

            state = next_state

            if done:
                break

        # Store episode results
        steps = len(episode_actions1)
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
 
        if isinstance(agent1, AwareHumanPTAgent):
            if not isinstance(agent2, AwareHumanPTAgent):
                agent1.opp_epsilon = agent2.epsilon

        if isinstance(agent2, AwareHumanPTAgent):
            if not isinstance(agent1, AwareHumanPTAgent):
                agent2.opp_epsilon = agent1.epsilon

        # Progress update
        if verbose and (episode + 1) % 100 == 0:
            print(f"  Episode {episode + 1}/{episodes}: "
                  f"Avg rewards = {avg_reward1:.3f}, {avg_reward2:.3f}")

    return results

def analyze_matchup(results, agent1, agent2, agent1_type, agent2_type, game_name, games_dict):
    """Comprehensive analysis of a matchup"""

    actions = games_dict[game_name]['actions']

    fig = plt.figure(figsize=(15, 10))

    # 1. Learning curves
    ax1 = plt.subplot(2, 3, 1)
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
    ax2 = plt.subplot(2, 3, 2)
    if results['actions1'] and results['actions2']:
        last_episode = -1
        final_rounds = 50

        actions1 = results['actions1'][last_episode][-final_rounds:]
        actions2 = results['actions2'][last_episode][-final_rounds:]

        ax2.plot(actions1, 'o-', markersize=4, label=f'{agent1_type}', alpha=0.7)
        ax2.plot(actions2, 's-', markersize=4, label=f'{agent2_type}', alpha=0.7)
        ax2.set_xlabel(f'Round (last {final_rounds})')
        ax2.set_ylabel('Action')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(actions)
        ax2.set_title('Final Strategy Pattern')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Action distribution
    ax3 = plt.subplot(2, 3, 3)
    if results['actions1']:
        # Last 10 episodes
        recent_actions1 = [a for ep in results['actions1'][-10:] for a in ep]
        recent_actions2 = [a for ep in results['actions2'][-10:] for a in ep]

        bins = np.arange(-0.5, 2.5, 1)
        ax3.hist([recent_actions1, recent_actions2], bins=bins,
                label=[f'{agent1_type}', f'{agent2_type}'],
                alpha=0.7, align='mid')
        ax3.set_xlabel('Action')
        ax3.set_ylabel('Frequency')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(actions)
        ax3.set_title('Action Distribution (last 10 episodes)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Strategy evolution (for aware PT)
    ax4 = plt.subplot(2, 3, 4)

    # 5. Cumulative rewards
    ax5 = plt.subplot(2, 3, 5)
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
    ax6 = plt.subplot(2, 3, 6)
    last_100_1 = results['avg_rewards1'][-100:] if len(results['avg_rewards1']) >= 100 else results['avg_rewards1']
    last_100_2 = results['avg_rewards2'][-100:] if len(results['avg_rewards2']) >= 100 else results['avg_rewards2']

    ax6.boxplot([last_100_1, last_100_2], labels=[agent1_type, agent2_type])
    ax6.set_ylabel('Average Reward')
    ax6.set_title('Reward Distribution (last 100 episodes)')
    ax6.grid(True, alpha=0.3)

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
        if results['actions1']:
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

def run_complete_experiment(game_name, payoff_matrix, episodes=300):
    """Run all agent matchups for a game"""

    print("\n" + "="*80)
    print(f"COMPLETE EXPERIMENT: {game_name}")
    print("="*80)

    # Standard PT parameters
    pt_params = {
        'lambd': 2.25,   # Loss aversion
        'alpha': 0.88,   # Diminishing sensitivity
        'gamma': 0.61,   # Probability weighting
        'r': 0           # Reference point
    }
  
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
            agent1 = LearningHumanPTAgent(env.state_size, action_size, action_size, pt_params, agent_id=0)
        elif agent1_type == 'AI':  # AI
            agent1 = AIAgent(env.state_size, action_size, agent_id=0)

        if agent2_type == 'LH':
            agent2 = LearningHumanPTAgent(env.state_size, action_size, action_size, pt_params, agent_id=1)
        elif agent2_type == 'AI':  # AI
            agent2 = AIAgent(env.state_size, action_size, agent_id=1)

        if agent1_type == 'Aware_PT':
            opp_params = dict()
            opp_params['opponent_action_size'] = action_size
            opp_params['q_values'] = agent2.q_values
            opp_params['opponent_type'] = agent2_type
            opp_params['epsilon'] = agent2.epsilon
            if agent2_type == 'LH':
                opp_params['beliefs'] = agent2.beliefs
                opp_params['tau'] = agent2.tau
                opp_params['temp'] = agent2.temperature

            agent1 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=0, opp_params=opp_params)

        if agent2_type == 'Aware_PT':
            opp_params = dict()
            opp_params['opponent_action_size'] = action_size
            opp_params['q_values'] = agent1.q_values
            opp_params['opponent_type'] = agent1_type
            opp_params['epsilon'] = agent1.epsilon
            if agent1_type == 'LH':
                opp_params['beliefs'] = agent1.beliefs
                opp_params['tau'] = agent1.tau
                opp_params['temp'] = agent1.temperature
            

            agent2 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=1, opp_params=opp_params)


        # Train the matchup
        print(f"Training {episodes} episodes...")
        results = train_agents(agent1, agent2, env, episodes=episodes, verbose=True)

        # Store results
        matchup_key = f"{agent1_type}_vs_{agent2_type}"
        all_results[matchup_key] = {
            'results': results,
            'agent1': agent1,
            'agent2': agent2
        }

        # Analyze this matchup
        games_dict = get_all_games()
        analyze_matchup(results, agent1, agent2, agent1_type, agent2_type, game_name, games_dict)

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
