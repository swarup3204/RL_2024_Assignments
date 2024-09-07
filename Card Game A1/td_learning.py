import matplotlib.pyplot as plt
import random
from collections import defaultdict
from copy import deepcopy
from CardGameEnv import CardGameEnv


def td_zero(env, num_episodes=10000, alpha=0.2, gamma=0.9):
    """
    TD(0) algorithm for value function estimation with reward tracking.
    """
    V = defaultdict(float)
    all_rewards = {}

    for episode in range(1, num_episodes + 1):
        seed = random.randint(0, num_episodes)
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

            V[state_tuple] += alpha * \
                (reward + gamma * V.get(new_state_tuple, 0) - V.get(state_tuple, 0))
            state = deepcopy(new_state)

        # Store rewards for each episode under its seed
        if all_rewards.get(seed) is None:
            all_rewards[seed] = [reward]
        else:
            all_rewards[seed].append(episode_reward)

    return V, all_rewards


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


def td_zero_main(n=8, num_episodes=100000, alpha=0.2, gamma=0.9):
    env = CardGameEnv(n)

    # Run TD(0) and get the value function and rewards for all episodes
    V, all_rewards = td_zero(
        env, num_episodes=num_episodes, alpha=alpha, gamma=gamma
    )

    print("Final Estimated Value Function:")
    for state, value in V.items():
        print(f"State: {state}, Value: {value}")

    # Plot rewards for the top 5 episodes with the maximum length
    plot_top_n_episodes_rewards(all_rewards, top_n=5)


if __name__ == "__main__":
    td_zero_main()
