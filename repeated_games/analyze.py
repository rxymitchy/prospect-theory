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

    fig = plt.figure(figsize=(15, 10))

    # 1. Avg rewards
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

    if len(results['ref_points1']) > 0:
        ref_points1 = results['ref_points1']
        ax2.plot(ref_points1, label=f'{agent1_type}')

    if len(results['ref_points2']) > 0:
        ref_points2 = results['ref_points2']
        ax2.plot(ref_points2, label=f'{agent2_type}')

    ax2.set_xlabel(f'Step (Every 100)')
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

    # 4. Q value convergence 
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

    # 6. Learning convergence (how much are q values changing)
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
    ax8 = plt.subplot(3, 3, 8)
    actions_1, actions_2 = results['actions1'], results['actions2']

    assert len(actions_1) == len(actions_2)

    action_list = [] # Maybe an inefficient way to track actions, but I'm here, so...

    plot_1, plot_2 = {}, {}
    for idx in range(len(actions_1)):
        # Count the frequency of each action
        ep_actions_1, ep_actions_2 = Counter(actions_1[idx]), Counter(actions_2[idx])

        # Update plot lists so that the lengths = idx, defaulted to 0
        for key, v in plot_1.items():
            v.append(0)
        for key, v in plot_2.items():
            v.append(0)

        # Go through each action frequency and append it to the plotting data
        for action, count in ep_actions_1.items():
            if action not in action_list:
                action_list.append(action)

            prob = count / len(actions_1[idx])
        
            # Update action tracker
            if str(action) not in plot_1.keys():
                plot_1[str(action)] = [0] * (idx + 1)
            plot_1[str(action)][idx] = prob

        for action, count in ep_actions_2.items():
            prob = count / len(actions_2[idx])
            if str(action) not in plot_2.keys():
                plot_2[str(action)] = [0] * (idx + 1)
            plot_2[str(action)][idx] = prob

    # smoothing logic
    window = len(actions_1) // 20
    x_labels = range(window-1, len(actions_1))

    for label, values in plot_1.items():
        smooth_values = smooth(values, window)
        ax7.plot(x_labels, smooth_values, label=f'Player 1 Action: {label}')

    for label, values in plot_2.items():
        smooth_values = smooth(values, window)
        ax8.plot(x_labels, smooth_values, label=f'Player 2 Action: {label}')
        
    ax7.set_title("Player Policies (Action Probs) Over Time")
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

