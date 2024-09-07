import matplotlib.pyplot as plt
import seaborn as sns
from value_iteration import value_iteration
from td_learning import td_zero
from CardGameEnv import CardGameEnv

# Initialize the environment
env = CardGameEnv(n=8)

# Get value functions and policies from Value Iteration and TD Learning
# Assuming value_iteration and td_zero return (values, policies, rewards) for each episode or iteration
vi_values_dict, vi_policies, vi_rewards = value_iteration(
    env)#, track_intermediate=True)
td_values_dict, td_policies, td_rewards = td_zero(env)#, track_intermediate=True)

# Define the states and indices for consistent plotting
states = sorted(vi_values_dict.keys())
state_indices = range(len(states))

# Helper function to flatten values for plotting


def flatten_values(value):
    if isinstance(value, list) or isinstance(value, tuple):
        return value[0]
    return value

# Heatmap function for visualizing policies


def plot_policy_heatmap(policy, title):
    flattened_policy = [flatten_values(policy[state]) for state in states]
    policy_matrix = [flattened_policy[i:i + env.n]
                     for i in range(0, len(flattened_policy), env.n)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(policy_matrix, annot=True,
                cmap="coolwarm", cbar=True, square=True)
    plt.title(title)
    plt.xlabel("State Index")
    plt.ylabel("Policy Actions")
    plt.show()

# Line plot for reward convergence


def plot_rewards(rewards_vi, rewards_td):
    plt.plot(rewards_vi, label="Value Iteration", marker='o')
    plt.plot(rewards_td, label="TD Learning", marker='x')
    plt.xlabel("Episode/Iteration")
    plt.ylabel("Total Reward")
    plt.title("Reward Convergence: Value Iteration vs TD Learning")
    plt.legend()
    plt.show()


# Initial, intermediate, and final policies (modify based on how many steps are tracked)
vi_initial_policy, vi_intermediate_policy, vi_final_policy = vi_policies[0], vi_policies[len(
    vi_policies)//2], vi_policies[-1]
td_initial_policy, td_intermediate_policy, td_final_policy = td_policies[0], td_policies[len(
    td_policies)//2], td_policies[-1]

# Plot policy heatmaps for Value Iteration
plot_policy_heatmap(vi_initial_policy, "Value Iteration - Initial Policy")
plot_policy_heatmap(vi_intermediate_policy,
                    "Value Iteration - Intermediate Policy")
plot_policy_heatmap(vi_final_policy, "Value Iteration - Final Policy")

# Plot policy heatmaps for TD Learning
plot_policy_heatmap(td_initial_policy, "TD Learning - Initial Policy")
plot_policy_heatmap(td_intermediate_policy,
                    "TD Learning - Intermediate Policy")
plot_policy_heatmap(td_final_policy, "TD Learning - Final Policy")

# Plot rewards for both algorithms
plot_rewards(vi_rewards, td_rewards)

# Plot value function convergence (final state values)
vi_values = [flatten_values(vi_values_dict[state]) for state in states]
td_values = [flatten_values(td_values_dict[state]) for state in states]

plt.plot(state_indices, vi_values, label="Value Iteration", marker='o')
plt.plot(state_indices, td_values, label="TD Learning", marker='x')
plt.xlabel("State Index")
plt.ylabel("State Values")
plt.legend()
plt.title("Comparison of Value Iteration and TD Learning State Values")
plt.show()
