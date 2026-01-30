import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from .aware_human import AwareHumanPTAgent
from .learning_human import LearningHumanPTAgent
from .ai_agent import AIAgent
from .fictitiousplay import FictitiousPlayAgent, SmoothFictitiousPlayAgent
from .game_env import RepeatedGameEnv
from .utils import get_all_games


def train_fictitious_play(agent1, agent2, env, episodes=500):
    """
    Train two fictitious play agents against each other
    """
    results = {
        'rewards1': [],
        'rewards2': [],
        'actions1': [],
        'actions2': [],
        'avg_rewards1': [],
        'avg_rewards2': []
    }
    
    for episode in range(episodes):
        # Reset agents for new episode
        agent1.reset()
        agent2.reset()
        
        episode_rewards1 = 0
        episode_rewards2 = 0
        episode_actions1 = []
        episode_actions2 = []
        
        # Get initial state (FP doesn't use state, but env needs it)
        state = env.reset()
        
        for _ in range(env.horizon):
            # Agents choose actions
            action1 = agent1.act(state)
            action2 = agent2.act(state)
            
            # Execute step
            next_state, reward1, reward2, done, _ = env.step(action1, action2)
            
            # Update FP agents
            agent1.update(action2, state)
            agent2.update(action1, state)
            
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
        results['rewards1'].append(episode_rewards1)
        results['rewards2'].append(episode_rewards2)
        results['actions1'].append(episode_actions1)
        results['actions2'].append(episode_actions2)
        results['avg_rewards1'].append(episode_rewards1 / steps)
        results['avg_rewards2'].append(episode_rewards2 / steps)
        
        # Progress update
        if (episode + 1) % 100 == 0:
            avg1 = episode_rewards1 / steps
            avg2 = episode_rewards2 / steps
            print(f"  FP Episode {episode + 1}/{episodes}: "
                  f"Avg rewards = {avg1:.3f}, {avg2:.3f}")
    
    return results


def compare_algorithms(game_name, payoff_matrix, episodes=300, pt_params={}):
    """
    Compare Algorithm 1 vs Fictitious Play baseline
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: {game_name}")
    print(f"{'='*80}")
    
    # Create environment
    env = RepeatedGameEnv(payoff_matrix, horizon=100, state_history=2)
    action_size = 2
    
    # Test configurations
    configs = [
        # (agent1_type, agent2_type, use_pt_in_fp)
        ('AI', 'AI', False),           # Baseline: both EU agents
        ('LH', 'LH', True),            # Both PT learning agents
        ('AI', 'LH', True),            # Mixed: EU vs PT
    ]
    
    all_results = {}
    
    for agent1_type, agent2_type, use_pt in configs:
        print(f"\nTesting: {agent1_type} vs {agent2_type}")
        print("-" * 40)
        
        # === ALGORITHM 1 ===
        print("Algorithm 1 (Our method):")
        
        # Create Algorithm 1 agents
        if agent1_type == 'AI':
            algo1_agent1 = AIAgent(env.state_size, action_size, agent_id=0)
        elif agent1_type == 'LH':
            algo1_agent1 = LearningHumanPTAgent(
                env.state_size, action_size, action_size, 
                pt_params, agent_id=0
            )
        
        if agent2_type == 'AI':
            algo1_agent2 = AIAgent(env.state_size, action_size, agent_id=1)
        elif agent2_type == 'LH':
            algo1_agent2 = LearningHumanPTAgent(
                env.state_size, action_size, action_size,
                pt_params, agent_id=1
            )
        
        # Train Algorithm 1
        from .train import train_agents
        algo1_results = train_agents(
            algo1_agent1, algo1_agent2, env, 
            episodes=episodes, verbose=False
        )
        
        # === FICTITIOUS PLAY BASELINE ===
        print("Fictitious Play (Baseline):")
        
        # Create FP agents
        fp_agent1 = FictitiousPlayAgent(
            payoff_matrix, agent_id=0, 
            use_pt=use_pt, pt_params=pt_params
        )
        fp_agent2 = FictitiousPlayAgent(
            payoff_matrix, agent_id=1,
            use_pt=use_pt, pt_params=pt_params
        )
        
        # Train FP
        fp_results = train_fictitious_play(
            fp_agent1, fp_agent2, env, 
            episodes=episodes
        )
        
        # Store results
        config_key = f"{agent1_type}_vs_{agent2_type}"
        all_results[config_key] = {
            'algo1': algo1_results,
            'fp': fp_results,
            'algo1_agents': (algo1_agent1, algo1_agent2),
            'fp_agents': (fp_agent1, fp_agent2)
        }
    
    return all_results


def plot_comparison(all_results, game_name):
    """
    Plot comparison between Algorithm 1 and Fictitious Play
    """
    n_configs = len(all_results)
    fig, axes = plt.subplots(n_configs, 2, figsize=(14, 5*n_configs))
    
    if n_configs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (config_key, data) in enumerate(all_results.items()):
        algo1_results = data['algo1']
        fp_results = data['fp']
        
        # Plot learning curves
        ax1 = axes[idx, 0]
        
        # Smooth Algorithm 1 rewards
        window = 20
        if len(algo1_results['avg_rewards1']) >= window:
            smooth_algo1 = np.convolve(
                algo1_results['avg_rewards1'], 
                np.ones(window)/window, 
                mode='valid'
            )
            ax1.plot(smooth_algo1, 'b-', label='Algorithm 1', linewidth=2)
        
        # Smooth FP rewards
        if len(fp_results['avg_rewards1']) >= window:
            smooth_fp = np.convolve(
                fp_results['avg_rewards1'],
                np.ones(window)/window,
                mode='valid'
            )
            ax1.plot(smooth_fp, 'r--', label='Fictitious Play', linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.set_title(f'{config_key} - Learning Curves\n{game_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot final performance comparison
        ax2 = axes[idx, 1]
        
        # Last 50 episodes average
        last_n = 50
        if len(algo1_results['avg_rewards1']) >= last_n:
            algo1_final = np.mean(algo1_results['avg_rewards1'][-last_n:])
            algo1_std = np.std(algo1_results['avg_rewards1'][-last_n:])
        else:
            algo1_final = np.mean(algo1_results['avg_rewards1'])
            algo1_std = np.std(algo1_results['avg_rewards1'])
        
        if len(fp_results['avg_rewards1']) >= last_n:
            fp_final = np.mean(fp_results['avg_rewards1'][-last_n:])
            fp_std = np.std(fp_results['avg_rewards1'][-last_n:])
        else:
            fp_final = np.mean(fp_results['avg_rewards1'])
            fp_std = np.std(fp_results['avg_rewards1'])
        
        # Bar plot
        x_pos = [0, 1]
        means = [algo1_final, fp_final]
        stds = [algo1_std, fp_std]
        
        ax2.bar(x_pos, means, yerr=stds, 
                capsize=10, alpha=0.7,
                color=['blue', 'red'])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Algorithm 1', 'Fictitious Play'])
        ax2.set_ylabel('Average Reward (last 50 episodes)')
        ax2.set_title(f'{config_key} - Final Performance\n{game_name}')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add text with improvement percentage
        if fp_final != 0:
            improvement = ((algo1_final - fp_final) / abs(fp_final)) * 100
            ax2.text(0.5, max(means)*0.9, 
                    f'Improvement: {improvement:+.1f}%',
                    ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print(f"SUMMARY COMPARISON: {game_name}")
    print("="*80)
    
    for config_key, data in all_results.items():
        print(f"\n{config_key}:")
        
        # Algorithm 1 stats
        algo1_rewards = data['algo1']['avg_rewards1']
        fp_rewards = data['fp']['avg_rewards1']
        
        if len(algo1_rewards) >= 50:
            algo1_mean = np.mean(algo1_rewards[-50:])
            algo1_std = np.std(algo1_rewards[-50:])
        else:
            algo1_mean = np.mean(algo1_rewards)
            algo1_std = np.std(algo1_rewards)
        
        if len(fp_rewards) >= 50:
            fp_mean = np.mean(fp_rewards[-50:])
            fp_std = np.std(fp_rewards[-50:])
        else:
            fp_mean = np.mean(fp_rewards)
            fp_std = np.std(fp_rewards)
        
        print(f"  Algorithm 1: {algo1_mean:.3f} ± {algo1_std:.3f}")
        print(f"  Fictitious Play: {fp_mean:.3f} ± {fp_std:.3f}")
        
        if fp_mean != 0:
            improvement = ((algo1_mean - fp_mean) / abs(fp_mean)) * 100
            print(f"  Relative improvement: {improvement:+.1f}%")
        
        # Convergence comparison
        if algo1_std < 0.1 and fp_std < 0.1:
            print("  ✓ Both algorithms converged")
        elif algo1_std < fp_std:
            print(f"  → Algorithm 1 more stable (std: {algo1_std:.3f} vs {fp_std:.3f})")
        else:
            print(f"  → Fictitious Play more stable (std: {fp_std:.3f} vs {algo1_std:.3f})")


def run_full_comparison():
    """Run comparison across all games"""
    games = get_all_games()
    
    # Select key games for comparison
    test_games = ['PrisonersDilemma', 'MatchingPennies', 
                  'BattleOfSexes', 'StagHunt', 'Chicken']
    
    all_game_results = {}
    
    for game_name in test_games:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {game_name}")
        print(f"{'='*80}")
        
        payoff_matrix = games[game_name]['payoffs']
        
        # Run comparison
        results = compare_algorithms(game_name, payoff_matrix, episodes=200)
        all_game_results[game_name] = results
        
        # Plot for this game
        plot_comparison(results, game_name)
    
    # Create cross-game summary
    print("\n" + "="*80)
    print("CROSS-GAME SUMMARY: Algorithm 1 vs Fictitious Play")
    print("="*80)
    
    summary_data = []
    
    for game_name in test_games:
        for config_key, data in all_game_results[game_name].items():
            # Calculate final performance
            algo1_rewards = data['algo1']['avg_rewards1']
            fp_rewards = data['fp']['avg_rewards1']
            
            if len(algo1_rewards) >= 50:
                algo1_mean = np.mean(algo1_rewards[-50:])
                fp_mean = np.mean(fp_rewards[-50:])
            else:
                algo1_mean = np.mean(algo1_rewards)
                fp_mean = np.mean(fp_rewards)
            
            improvement = ((algo1_mean - fp_mean) / max(abs(fp_mean), 0.001)) * 100
            
            summary_data.append({
                'Game': game_name,
                'Matchup': config_key,
                'Algorithm1': algo1_mean,
                'FictitiousPlay': fp_mean,
                'Improvement%': improvement
            })
    
    # Print summary table
    print("\nPerformance Summary (Average Reward):")
    print("-" * 80)
    print(f"{'Game':20} {'Matchup':15} {'Algo1':>8} {'FP':>8} {'Improv%':>10}")
    print("-" * 80)
    
    for row in summary_data:
        print(f"{row['Game']:20} {row['Matchup']:15} "
              f"{row['Algorithm1']:8.3f} {row['FictitiousPlay']:8.3f} "
              f"{row['Improvement%']:10.1f}")
    
    return all_game_results


if __name__ == "__main__":
    run_full_comparison()
