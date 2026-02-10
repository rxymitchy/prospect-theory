import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from matplotlib.ticker import FuncFormatter

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
    MAX_ACTIONS = 3

    if joint_actions.shape[0] <= MAX_ACTIONS and joint_actions.shape[1] <= MAX_ACTIONS:
        for i in range(joint_actions.shape[0]):
            for j in range(joint_actions.shape[1]):
                ax3.text(
                    j, i,
                    int(joint_actions[i, j]),
                    ha="center",
                    va="center",
                    color="white" if joint_actions[i, j] > joint_actions.max() / 2 else "black"
                )

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
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Mean Absolute Error (|Q-R|)')
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

    if game_name == 'PrisonersDilemma': # PT NE == PT EB
        optimal_policies = [(0, 1), (0, 1)]
    elif game_name == 'MatchingPennies': # PT NE == PT EB
        optimal_policies = [(0.5, 0.5), (0.5, 0.5)]

    elif game_name == 'OchsGame':
        pass


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

