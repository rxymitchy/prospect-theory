import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from repeated_games.train import run_complete_experiment, compare_all_results, train_agents 

from repeated_games import (
    RepeatedGameEnv,
    AIAgent,
    ProspectTheory,
    LearningHumanPTAgent,
    AwareHumanPTAgent,
    DoubleAuction,
    train_agents,
    analyze_matchup,
    compare_all_results,
    get_all_games,
)
     

def interactive_experiment():

    print("="*80)
    print("PROSPECT THEORY REPEATED GAMES - ENHANCED EXPERIMENTS")
    print("="*80)

    # Check for PyTorch
    try:
        import torch
        print("✓ PyTorch is available")
    except ImportError:
        raise ImportError("Pytorch is not available")

    # Get games
    games = get_all_games()

    while True:
        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print("\nOptions:")
        print("1. Run quick demonstration (fast)")
        print("2. Run complete experiment for a specific game")
        print("3. Compare all games (summary)")
        print("4. Run custom matchup")
        print("5. Run Algorithm 1 vs Fictitious Play comparison")  
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        print("\n" + "="*80)
        print("SET REFERENCE POINT SETTING")
        print("="*80)
        print("\nOptions:")
        print("1. Fixed (defaults at 0, go into the code to change")
        print("2. EMA")
        print("3. Q (updates ref point based on max(Q(S, A)) at the current state. Normalized with (1-gamma) to move from expected discounted return scale to reward scale. ")
        print("4. EMAOR (Exponential Moving Average over Opponent Rewards, basically tracks how well the opponent is doing not how well the player is doing")
      
        ref_setting = ""
        viable_options = ['Fixed', 'EMA', 'Q', 'EMAOR']

        while ref_setting not in viable_options:        
            ref_setting = input("\nEnter choice (type exactly as written, defaults to fixed): 'Fixed', 'EMA', 'Q', 'EMAOR': ").strip()

            if ref_setting not in viable_options:
                print("Failed, please try again.")


        if choice == '1':
            # Quick demo
            # Im not updating this branch with the new logic so it will fail.
            print("\nRunning quick demonstration with Prisoner's Dilemma...")
            game_name = 'PrisonersDilemma'
            payoff_matrix = games[game_name]['payoffs']

            # Run a quick version
            env = RepeatedGameEnv(payoff_matrix, horizon=50, state_history=1)
            pt_params = {'lambd': 2.25, 'alpha': 0.88, 'gamma': 0.61, 'r': 0}

            # Create agents
            action_size = 2
            agent1 = LearningHumanPTAgent(env.state_size, action_size, action_size, pt_params, agent_id=0, ref_setting=ref_setting)
            agent2 = AIAgent(env.state_size, 2, 1)

            print("\nTraining Aware PT vs AI for 100 episodes...")
            results = train_agents(agent1, agent2, env, episodes=100, verbose=False)

            # Quick analysis
            final_avg1 = np.mean(results['avg_rewards1'][-20:]) if results['avg_rewards1'] else 0
            final_avg2 = np.mean(results['avg_rewards2'][-20:]) if results['avg_rewards2'] else 0

            print(f"\nQuick Results:")
            print(f"  Aware PT Agent: {final_avg1:.3f}")
            print(f"  AI Agent: {final_avg2:.3f}")

            # Simple plot
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(results['avg_rewards1'], label='Aware PT')
            plt.plot(results['avg_rewards2'], label='AI')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.title('Learning Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            if results['actions1']:
                last_actions = results['actions1'][-1][-20:]
                plt.plot(last_actions, 'o-', label='Aware PT Actions')
                plt.xlabel('Round (last 20)')
                plt.ylabel('Action')
                plt.title('Final Strategy Pattern')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yticks([0, 1])

            plt.tight_layout()
            plt.show()

        elif choice == '2':
            # Complete experiment for specific game
            print("\nAvailable games:")
            for i, (name, data) in enumerate(games.items(), 1):
                print(f"{i}. {name}: {data['description']}")

            game_choice = input("\nEnter game number: ").strip()
            game_names = list(games.keys())

            if game_choice.isdigit() and 1 <= int(game_choice) <= len(games):
                game_name = game_names[int(game_choice)-1]
                payoff_matrix = games[game_name]['payoffs']

                episodes = input(f"Episodes to run (default 200): ").strip()
                episodes = int(episodes) if episodes.isdigit() else 200

                print(f"\nStarting complete experiment for {game_name}...")
                all_results = run_complete_experiment(game_name, payoff_matrix, episodes=episodes, ref_setting=ref_setting)

                # Compare results
                compare_all_results(all_results, game_name)
            else:
                print("Invalid choice, using Prisoner's Dilemma")
                game_name = 'PrisonersDilemma'
                payoff_matrix = games[game_name]['payoffs']
                all_results = run_complete_experiment(game_name, payoff_matrix, ref_setting=ref_setting)
                compare_all_results(all_results, game_name)

        elif choice == '3':
            # Compare all games
            print("\nRunning summary comparison across all games...")

            summary_data = []
            test_games = ['PrisonersDilemma', 'MatchingPennies', 'BattleOfSexes', 'StagHunt', 'Chicken']

            action_size = 2

            for game_name in test_games:
                print(f"\nAnalyzing {game_name}...")
                payoff_matrix = games[game_name]['payoffs']

                # Run quick version
                env = RepeatedGameEnv(payoff_matrix, horizon=50, state_history=2)
                pt_params = {'lambd': 2.25, 'alpha': 0.88, 'gamma': 0.61, 'r': 0}

                # Test one key matchup
                agent1 = LearningHumanPTAgent(env.state_size, action_size, action_size, pt_params, agent_id=0, ref_setting=ref_setting)
                agent2 = AIAgent(env.state_size, 2, 2, 1)

                results = train_agents(agent1, agent2, env, episodes=100, verbose=False)

                if results['avg_rewards1'] and len(results['avg_rewards1']) >= 20:
                    final_avg1 = np.mean(results['avg_rewards1'][-20:])
                    final_avg2 = np.mean(results['avg_rewards2'][-20:])

                    summary_data.append({
                        'Game': game_name,
                        'Aware_PT_Avg': final_avg1,
                        'AI_Avg': final_avg2,
                        'Difference': final_avg1 - final_avg2,
                        'Description': games[game_name]['description']
                    })

            # Display summary
            if summary_data:
                df = pd.DataFrame(summary_data)
                print("\n" + "="*80)
                print("SUMMARY COMPARISON ACROSS GAMES")
                print("="*80)
                print(df.to_string(index=False))

                # Visualization
                fig, ax = plt.subplots(figsize=(12, 6))

                x = np.arange(len(df))
                width = 0.35

                ax.bar(x - width/2, df['Aware_PT_Avg'], width, label='Aware PT', alpha=0.7)
                ax.bar(x + width/2, df['AI_Avg'], width, label='AI', alpha=0.7)

                ax.set_xlabel('Game')
                ax.set_ylabel('Average Reward')
                ax.set_title('Aware PT vs AI Performance Across Games')
                ax.set_xticks(x)
                ax.set_xticklabels(df['Game'], rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                plt.show()

        elif choice == '4':
            # Custom matchup
            print("\nCustom Matchup Configuration")
            print("-"*40)

            # Select game
            print("\nSelect game:")
            for i, (name, data) in enumerate(games.items(), 1):
                print(f"{i}. {name}")

            game_choice = input("\nGame number: ").strip()
            game_names = list(games.keys())

            if game_choice.isdigit() and 1 <= int(game_choice) <= len(games):
                game_name = game_names[int(game_choice)-1]
                payoff_matrix = games[game_name]['payoffs']
            else:
                game_name = 'PrisonersDilemma'
                payoff_matrix = games[game_name]['payoffs']

            # Select agent types
            print("\nAgent types: Aware_PT, Learning_PT, AI")
            agent1_type = input("Agent 1 type: ").strip()
            agent2_type = input("Agent 2 type: ").strip()

            # Validate agent types
            valid_types = ['Aware_PT', 'Learning_PT', 'AI']
            agent1_type = agent1_type if agent1_type in valid_types else 'Aware_PT'
            agent2_type = agent2_type if agent2_type in valid_types else 'AI'

            # Get parameters
            episodes = input("Episodes (default 200): ").strip()
            episodes = int(episodes) if episodes.isdigit() else 200

            # Run custom matchup
            print(f"\nRunning {agent1_type} vs {agent2_type} in {game_name}...")

            env = RepeatedGameEnv(payoff_matrix, horizon=100, state_history=2)
            pt_params = {'lambd': 2.25, 'alpha': 0.88, 'gamma': 0.61, 'r': 0}

            # Reference point setting
            # Options = Fixed, EMA, Q, EMAOR
            ref_lambda = 0.9 

            # Create agents
            action_size = 2
            if agent1_type == 'Aware_PT':
                opp_params = dict()
                opp_params['opponent_type'] = agent2_type
                opp_params['opponent_action_size'] = action_size
                opp_params['opp_ref'] = None
                if agent2_type != "AI":
                    opp_params['opp_ref'] = agent2.ref_point
                agent1 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=0, opp_params=opp_params, ref_setting=ref_setting, lambda_ref = ref_lambda)
            elif agent1_type == 'Learning_PT':
                agent1 = LearningHumanPTAgent(env.state_size, 2, 2, pt_params, 0, ref_setting=ref_setting, lambda_ref = ref_lambda)
            else:
                agent1 = AIAgent(env.state_size, 2, 2, 0)

            if agent2_type == 'Aware_PT':
                opp_params = dict()
                opp_params['opponent_type'] = agent1_type
                opp_params['opponent_action_size'] = action_size
                opp_params['opp_ref'] = None
                if agent1_type != "AI":
                    opp_params['opp_ref'] = agent1.ref_point
                agent2 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=1,opp_params=opp_params, ref_setting=ref_setting, lambda_ref = ref_lambda)
            elif agent2_type == 'Learning_PT':
                agent2 = LearningHumanPTAgent(env.state_size, 2, 2, pt_params, 1, ref_setting=ref_setting, lambda_ref = ref_lambda)
            else:
                agent2 = AIAgent(env.state_size, 2, 2, 1)

            # Train
            results = train_agents(agent1, agent2, env, episodes=episodes, verbose=True)

            # Analyze
            games_dict = get_all_games()
            print(payoff_matrix)
            if agent1_type != 'Aware_PT':
                agent1_q_vals = agent1.get_q_values()
                print(f"Agent 1 state visits: {agent1.state_visit_counter}")
                print(f"Agent 1 raw q values = {agent1_q_vals}, agent 1 normalized q values = {(1-agent1.gamma) * agent1_q_vals}")

            if agent2_type != 'Aware_PT':
                agent2_q_vals = agent2.get_q_values()
                print(f"Agent 2 raw q values = {agent2_q_vals}, agent 2 normalized q values = {(1-agent2.gamma) * agent2_q_vals}")

                print(f"Agent 2 state visits: {agent2.state_visit_counter}")


            if hasattr(agent1, "beliefs"):
                print(f"Agent 1 beliefs: {agent1.beliefs}")

            if hasattr(agent2, "beliefs"):
                print(f"Agent 2 beliefs: {agent2.beliefs}")

            analyze_matchup(results, agent1, agent2, agent1_type, agent2_type, game_name, games_dict, payoff_matrix)

        elif choice == '5':
            # Algorithm 1 vs Fictitious Play comparison
            print("\nRunning Algorithm 1 vs Fictitious Play comparison...")
            
            # Select game
            print("\nAvailable games:")
            for i, (name, data) in enumerate(games.items(), 1):
                print(f"{i}. {name}: {data['description']}")
            
            game_choice = input("\nEnter game number: ").strip()
            game_names = list(games.keys())
            
            if game_choice.isdigit() and 1 <= int(game_choice) <= len(games):
                game_name = game_names[int(game_choice)-1]
                payoff_matrix = games[game_name]['payoffs']
                
                episodes = input(f"Episodes to run (default 200): ").strip()
                episodes = int(episodes) if episodes.isdigit() else 200
                
                # Run comparison
                from repeated_games.baseline import compare_algorithms, plot_comparison
                
                print(f"\nComparing Algorithm 1 vs Fictitious Play in {game_name}...")
                results = compare_algorithms(game_name, payoff_matrix, episodes=episodes)
                plot_comparison(results, game_name)
                
            else:
                print("Invalid choice, using Prisoner's Dilemma")
                game_name = 'PrisonersDilemma'
                payoff_matrix = games[game_name]['payoffs']
                from repeated_games.baseline import compare_algorithms, plot_comparison
                results = compare_algorithms(game_name, payoff_matrix, episodes=200)
                plot_comparison(results, game_name)

        elif choice == '6':
            print("\nExiting...")
            break

        else:
            print("\nInvalid choice. Please try again.")

        # Ask to continue
        cont = input("\nContinue? (y/n): ").strip().lower()
        if cont != 'y':
            print("\nExiting...")
            break


if __name__ == "__main__":
    # Run the interactive interface
    interactive_experiment()
