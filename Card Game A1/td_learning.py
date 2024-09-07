# Roll: 20CS30062
# CAV1


import matplotlib.pyplot as plt
import random
from collections import defaultdict
from copy import deepcopy
from CardGameEnv import CardGameEnv


def td_zero(env, num_episodes=10000, alpha=0.2, gamma=0.9, n_tracked_states=10):
    """
    TD(0) algorithm for value function estimation with reward and value tracking.
    """
    V = defaultdict(float)
    # Track value estimates for the tracked states
    tracked_values = defaultdict(list)
    all_rewards = {}  # Track total rewards for each episode

    # Select n_tracked_states randomly from all possible states after the first episode
    random_states = set()

    for episode in range(1, num_episodes + 1):
        seed = random.randint(0, num_episodes//4)
        env.seed(seed)
        state = deepcopy(env.reset())
        done = False
        episode_reward = 0

        while not done:
            agent_deck = state['agent_deck']
            state_tuple = (tuple(state['agent_deck']),
                           state['opponent_card_shown'])

            action = random.randint(0, len(agent_deck) - 1)
            new_state, reward, done = env.step(action)
            episode_reward += reward

            new_state_tuple = (
                tuple(new_state['agent_deck']), new_state['opponent_card_shown'])

            if done:
                V[new_state_tuple] = 0

            # Update value function
            V[state_tuple] += alpha * \
                (reward + gamma * V.get(new_state_tuple, 0) - V.get(state_tuple, 0))
            state = deepcopy(new_state)

        # Randomly select states after the first episode
        if episode == 100:
            all_states = list(V.keys())
            random_states = set(random.sample(all_states, n_tracked_states))

        # Track the values of the selected states
        for state in random_states:
            tracked_values[state].append(V.get(state, 0))

        # Store rewards for each episode under its seed
        if all_rewards.get(seed) is None:
            all_rewards[seed] = [episode_reward]
        else:
            all_rewards[seed].append(episode_reward)

    return V, tracked_values, all_rewards


def plot_value_convergence(tracked_values):
    """
    Plot the value convergence for the tracked states over the episodes.
    """
    plt.figure(figsize=(12, 8))

    for state, values in tracked_values.items():
        plt.plot(values, label=f'State {state}')

    plt.title(f"Value Convergence for Randomly Selected States")
    plt.xlabel("Episode")
    plt.ylabel("Estimated Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_top_n_episodes_rewards(all_rewards, top_n=5):
    """
    Plot the rewards of the top N episodes with the maximum length of rewards.
    """
    # Sort episodes by the length of rewards in descending order
    sorted_episodes = sorted(
        all_rewards.items(), key=lambda x: len(x[1]), reverse=True)

    # Select the top N episodes with the longest lengths
    top_episodes = sorted_episodes[:top_n]

    plt.figure(figsize=(12, 8))

    # Plot rewards for each of the top N episodes
    for i, (seed, rewards) in enumerate(top_episodes):
        plt.plot(range(len(rewards)), rewards, marker='o', linestyle='-',
                 label=f'Episode {seed} (Length: {len(rewards)})')

    plt.title(f"Rewards Over Time for Top {top_n} Longest Episodes")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def td_zero_main(n=20, num_episodes=400000, alpha=0.2, gamma=0.9, n_tracked_states=10):
    env = CardGameEnv(n)

    # Run TD(0) with value and reward tracking
    V, tracked_values, all_rewards = td_zero(
        env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, n_tracked_states=n_tracked_states
    )

    print("Final Estimated Value Function:")
    for state, value in V.items():
        print(f"State: {state}, Value: {value}")

    # Plot value convergence for the randomly selected states
    plot_value_convergence(tracked_values)

    # Plot rewards for the top 5 episodes with the maximum length
    plot_top_n_episodes_rewards(all_rewards, top_n=5)


if __name__ == "__main__":
    td_zero_main()
