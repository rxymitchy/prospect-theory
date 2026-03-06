import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from collections import Counter
from matplotlib.ticker import FuncFormatter
from .utils import smooth

def analyze_matchup(results, agent1, agent2, agent1_type, agent2_type, game_name, games_dict, payoff_matrix, pt_params):
    """
    I am not going through in line, and i went a little in depth on the read me. 
    I feel like this is pretty straightforward, but if its not please just send me an email,
    I'll be quick to respond
    """
    num_experiments = len(results.keys())

    fig = plt.figure(figsize=(15, 10))

    # 1. Avg rewards
    ax1 = plt.subplot(3, 3, 1)
    window = 20

    smoothed_p1 = []
    smoothed_p2 = []
    for idx in range(len(results.keys())):
        smoothed1 = np.convolve(results[f"{idx}"]['avg_rewards1'], np.ones(window)/window, mode='valid')
        smoothed2 = np.convolve(results[f"{idx}"]['avg_rewards2'], np.ones(window)/window, mode='valid')

        smoothed_p1.append(smoothed1)
        smoothed_p2.append(smoothed2)
        
    smoothed_p1 = np.stack(smoothed_p1)
    smoothed_p2 = np.stack(smoothed_p2)

    mean_p1, mean_p2 = np.mean(smoothed_p1, axis=0), np.mean(smoothed_p2, axis=0)
    std_p1, std_p2 = np.std(smoothed_p1, axis=0), np.std(smoothed_p2, axis=0)
    se_p1, se_p2 = std_p1 / np.sqrt(num_experiments), std_p2 / np.sqrt(num_experiments)

    ax1.plot(mean_p1, label=f'{agent1_type}', linewidth=2)
    ax1.plot(mean_p2, label=f'{agent2_type}', linewidth=2)

    x = np.arange(len(mean_p1))

    ax1.fill_between(x, mean_p1 + 1.96 * se_p1, mean_p1 - 1.96 * se_p1, alpha=0.3)
    ax1.fill_between(x, mean_p2 + 1.96 * se_p2, mean_p2 - 1.96 * se_p2, alpha=0.3)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward (± std)')
    ax1.set_title(f'95% Conf. Interval Reward Over {len(results.keys())} Runs\n{agent1_type} vs {agent2_type}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final strategy patterns
    ax2 = plt.subplot(3, 3, 2)

    ref_points_p1 = []
    ref_points_p2 = []

    for idx in range(len(results.keys())):
        ref_points1 = results[f"{idx}"]['ref_points1']
        print("Ref Points 1: ", ref_points1)
        ref_points2 = results[f"{idx}"]['ref_points2']

        ref_points_p1.append(ref_points1)
        ref_points_p2.append(ref_points2)

    ref_points_p1 = np.stack(ref_points_p1)
    ref_points_p2 = np.stack(ref_points_p2)

    mean_p1, mean_p2 = np.mean(ref_points_p1, axis=0), np.mean(ref_points_p2, axis=0)
    std_p1, std_p2 = np.std(ref_points_p1, axis=0), np.std(ref_points_p2, axis=0)
    se_p1, se_p2 = std_p1 / np.sqrt(num_experiments), std_p2 / np.sqrt(num_experiments)
    
    ax2.plot(mean_p1, label=f'{agent1_type}')
    ax2.plot(mean_p2, label=f'{agent2_type}')

    x = np.arange(len(mean_p1))

    ax2.fill_between(x, mean_p1 + 1.96 * se_p1, mean_p1 - 1.96 * se_p1, alpha=0.3)

    x = np.arange(len(mean_p2))
    ax2.fill_between(x, mean_p2 + 1.96 * se_p2, mean_p2 - 1.96 * se_p2, alpha=0.3)

    ax2.set_xlabel(f'Step (Every 100)')
    ax2.set_ylabel('Reference Point')
    ax2.set_title(f'95% Conf. Interval of Ref. Points Over {len(results.keys())} Runs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Action distribution
    ax3 = plt.subplot(3, 3, 3)

    joint_actions = []
    for idx in range(len(results.keys())):
        joint_action = results[f"{idx}"]["joint_actions"]
        joint_actions.append(joint_action)
      
    joint_actions = np.stack(joint_actions)

    mean_joint_actions = np.mean(joint_actions, axis=0)
    im = ax3.imshow(mean_joint_actions, aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax3, label='Count')
    MAX_ACTIONS = 3

    if mean_joint_actions.shape[0] <= MAX_ACTIONS and mean_joint_actions.shape[1] <= MAX_ACTIONS:
        for i in range(mean_joint_actions.shape[0]):
            for j in range(mean_joint_actions.shape[1]):
                ax3.text(
                    j, i,
                    int(mean_joint_actions[i, j]),
                    ha="center",
                    va="center",
                    color="white" if mean_joint_actions[i, j] > mean_joint_actions.max() / 2 else "black"
                )

    ax3.set_xticks(np.arange(mean_joint_actions.shape[1]))
    ax3.set_xticklabels(np.arange(1, mean_joint_actions.shape[1] + 1))

    ax3.set_yticks(np.arange(mean_joint_actions.shape[0]))
    ax3.set_yticklabels(np.arange(1, mean_joint_actions.shape[0] + 1))

    if game_name == 'Double Auction Game':
        ax3.set_xlabel(f'Seller ({agent2_type})')
        ax3.set_ylabel(f'Buyer ({agent1_type})')

    else:
        ax3.set_xlabel(f'{agent2_type}')
        ax3.set_ylabel(f'{agent1_type}')

    ax3.set_title(f"Mean Joint Action Heatmap Over {len(results.keys())} Runs")

    # 4. Q value convergence 
    # 4. Q value convergence - Player 1
    ax4 = plt.subplot(3, 3, 4)
    if len(results["0"]['q_values1']) > 0:
        q_values_p1 = []
        for idx in range(len(results.keys())):
            q_values1 = np.stack(results[f"{idx}"]['q_values1'])
            q_values_p1.append(q_values1)

        q_values_p1 = np.stack(q_values_p1)

        mean_q_p1 = np.mean(q_values_p1, axis=0)
        se_q_p1 = np.std(q_values_p1, axis=0) / np.sqrt(num_experiments)

        ax4.plot(mean_q_p1[:, 0, 0], label='Action 0', linewidth=1)
        ax4.plot(mean_q_p1[:, 1, 0], label='Action 1', linewidth=1)

        x = np.arange(len(mean_q_p1))

        ax4.fill_between(x, mean_q_p1[:, 0, 0] + 1.96 * se_q_p1[:, 0, 0], mean_q_p1[:, 0, 0] - 1.96 * se_q_p1[:, 0, 0], alpha=0.3)
        ax4.fill_between(x, mean_q_p1[:, 1, 0] + 1.96 * se_q_p1[:, 1, 0], mean_q_p1[:, 1, 0] - 1.96 * se_q_p1[:, 1, 0], alpha=0.3)

        opp_freq = mean_joint_actions.sum(axis=0) / mean_joint_actions.sum()
        expected_payoff_a0 = payoff_matrix[0, :, 0] @ opp_freq
        expected_payoff_a1 = payoff_matrix[1, :, 0] @ opp_freq
        ax4.axhline(expected_payoff_a0, linestyle="--", color='gray', alpha=0.5, label="Expected R(a0)")
        ax4.axhline(expected_payoff_a1, linestyle="--", color='black', alpha=0.5, label="Expected R(a1)")

    ax4.set_xlabel('Step')
    ax4.set_ylabel('Normalized Q-value')
    ax4.set_title(f'95% Conf. Interval {agent1_type} Q-Values Over {len(results.keys())} Runs')
    ax4.legend()

    # 5. Q value convergence - Player 2
    ax5 = plt.subplot(3, 3, 5)

    if len(results["0"]['q_values2']) > 0:
        q_values_p2 = []
        for idx in range(len(results.keys())):
            q_values2 = np.stack(results[f"{idx}"]['q_values2'])
            q_values_p2.append(q_values2)
            
        q_values_p2 = np.stack(q_values_p2)

        mean_q_p2 = np.mean(q_values_p2, axis=0)
        se_q_p2 = np.std(q_values_p2, axis=0) / np.sqrt(num_experiments)

        x = np.arange(len(mean_q_p2))
        
        ax5.plot(mean_q_p2[:, 0, 0], label='Action 0', linewidth=1)
        ax5.plot(mean_q_p2[:, 1, 0], label='Action 1', linewidth=1)
        ax5.fill_between(x, mean_q_p2[:, 0, 0] + 1.96 * se_q_p2[:, 0, 0], mean_q_p2[:, 0, 0] - 1.96 * se_q_p2[:, 0, 0], alpha=0.3)
        ax5.fill_between(x, mean_q_p2[:, 1, 0] + 1.96 * se_q_p2[:, 1, 0], mean_q_p2[:, 1, 0] - 1.96 * se_q_p2[:, 1, 0], alpha=0.3)

        opp_freq = mean_joint_actions.sum(axis=0) / mean_joint_actions.sum()
        expected_payoff_a0 = payoff_matrix[:, 0, 1] @ opp_freq
        expected_payoff_a1 = payoff_matrix[:, 1, 1] @ opp_freq
        ax5.axhline(expected_payoff_a0, linestyle="--", color='gray', alpha=0.5, label="Expected R(a0)")
        ax5.axhline(expected_payoff_a1, linestyle="--", color='black', alpha=0.5, label="Expected R(a1)")

    ax5.set_xlabel('Step')
    ax5.set_ylabel('Normalized Q-value')
    ax5.set_title(f'95% Conf. Interval {agent2_type} Q-Values Over {len(results.keys())} Runs')
    ax5.legend()

    # 6. Learning convergence (how much are q values changing)
    ax6 = plt.subplot(3, 3, 6)
    if game_name != 'Double Auction Game':
        k = 100
    else:
        k = 1

    if len(results["0"]['q_values1']) > 0:
        q_changes_p1 = []
        for idx in range(len(results.keys())):
            q_values1 = np.stack(results[f"{idx}"]['q_values1'])
            q_values1, q_values1_copy = q_values1[1:], q_values1[:-1]
            q_values1_diff = q_values1 - q_values1_copy
            q_change1 = np.mean(np.abs(q_values1_diff), axis = tuple(range(1, q_values1_diff.ndim)))

            q_changes_p1.append(q_change1)
                
        q_changes_p1 = np.stack(q_changes_p1)
        
        mean_changes_p1 = np.mean(q_changes_p1, axis=0) 
        se_changes_p1 = np.std(q_changes_p1, axis=0) / np.sqrt(num_experiments)

        x = np.arange(len(mean_changes_p1))

        ax6.plot(mean_changes_p1, label=f'{agent1_type}')
        ax6.fill_between(x, mean_changes_p1 + 1.96 * se_changes_p1, mean_changes_p1 - 1.96 * se_changes_p1, alpha=0.3)

    if len(results["0"]['q_values2']) > 0:
        q_changes_p2 = []
        for idx in range(len(results.keys())):
            q_values2 = np.stack(results[f"{idx}"]['q_values2'])
            q_values2, q_values2_copy = q_values2[1:], q_values2[:-1]
            q_values2_diff = q_values2 - q_values2_copy
            q_change2 = np.mean(np.abs(q_values2_diff), axis = tuple(range(1, q_values2_diff.ndim)))
            q_changes_p2.append(q_change2)
      

        q_changes_p2 = np.stack(q_changes_p2)
            
        mean_changes_p2 = np.mean(q_changes_p2, axis=0)
        se_changes_p2 = np.std(q_changes_p2, axis=0) / np.sqrt(num_experiments)

        x = np.arange(len(mean_changes_p2))

        ax6.plot(mean_changes_p2, label=f'{agent2_type}')
        ax6.fill_between(x, mean_changes_p2 + 1.96 * se_changes_p2, mean_changes_p2 - 1.96 * se_changes_p2, alpha=0.3)

    ax6.set_xlabel("Steps (every 100)")
    ax6.set_ylabel('Q Value Diff')
    ax6.set_title(f'95% Conf. Interval of Q Values Changes Over {len(results.keys())} Runs')
    ax6.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x*k)}")
    )
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 3, 7)
    ax8 = plt.subplot(3, 3, 8)

    num_experiments = len(results.keys())

    # First pass: collect per-experiment action prob curves
    # Structure: {action_str: [exp0_curve, exp1_curve, ...]} for each player
    all_plot_1 = {}
    all_plot_2 = {}

    for exp_idx in range(num_experiments):
        exp = results[str(exp_idx)]
        actions_1, actions_2 = exp['actions1'], exp['actions2']
        assert len(actions_1) == len(actions_2)

        plot_1, plot_2 = {}, {}
        for idx in range(len(actions_1)):
            ep_actions_1, ep_actions_2 = Counter(actions_1[idx]), Counter(actions_2[idx])

            for key, v in plot_1.items():
                v.append(0)
            for key, v in plot_2.items():
                v.append(0)

            for action, count in ep_actions_1.items():
                prob = count / len(actions_1[idx])
                if str(action) not in plot_1:
                    plot_1[str(action)] = [0] * (idx + 1)
                plot_1[str(action)][idx] = prob

            for action, count in ep_actions_2.items():
                prob = count / len(actions_2[idx])
                if str(action) not in plot_2:
                    plot_2[str(action)] = [0] * (idx + 1)
                plot_2[str(action)][idx] = prob

        # Accumulate into all_plot dicts
        for action, curve in plot_1.items():
            if action not in all_plot_1:
                all_plot_1[action] = []
            all_plot_1[action].append(curve)

        for action, curve in plot_2.items():
            if action not in all_plot_2:
                all_plot_2[action] = []
            all_plot_2[action].append(curve)

    # Second pass: ensemble and plot
    first_exp = results["0"]
    num_episodes = len(first_exp['actions1'])
    window = num_episodes // 20
    x_labels = range(window - 1, num_episodes)

    for action, curves in all_plot_1.items():
        smoothed = np.array([smooth(c, window) for c in curves])
        mean_curve = smoothed.mean(axis=0)
        std_curve = smoothed.std(axis=0)
        se = std_curve / np.sqrt(num_experiments)
        ax7.plot(x_labels, mean_curve, label=f'Action: {action}')
        ax7.fill_between(x_labels, mean_curve - 1.96 * se, mean_curve + 1.96 * se, alpha=0.3)

    for action, curves in all_plot_2.items():
        smoothed = np.array([smooth(c, window) for c in curves])
        mean_curve = smoothed.mean(axis=0)
        std_curve = smoothed.std(axis=0)
        se = std_curve / np.sqrt(num_experiments)
        ax8.plot(x_labels, mean_curve, label=f'Action: {action}')
        ax8.fill_between(x_labels, mean_curve - 1.96 * se, mean_curve + 1.96 * se, alpha=0.3)

    ax7.set_title("Player 1 Policies (Action Probs) Over Time")
    ax7.set_xlabel("Episodes")
    ax7.set_ylabel("Action Probabilities")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    ax8.set_title("Player 2 Policies (Action Probs) Over Time")
    ax8.set_xlabel("Episodes")
    ax8.set_ylabel("Action Probabilities")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    if game_name == 'PrisonersDilemma': # PT NE == PT EB
        optimal_policies = [(0, 1), (0, 1)]
    elif game_name == 'MatchingPennies': # PT NE == PT EB
        optimal_policies = [(0.5, 0.5), (0.5, 0.5)]

    elif game_name == 'OchsGame':
        pass


    plt.suptitle(f'{game_name}: {agent1_type} vs {agent2_type}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

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

