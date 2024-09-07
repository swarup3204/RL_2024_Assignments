from collections import defaultdict
from copy import deepcopy
from CardGameEnv import CardGameEnv
import random
import matplotlib.pyplot as plt
import seaborn as sns


def td_zero(env, num_episodes=10000, alpha=0.2, gamma=1.0, heatmap_interval=2000):
    """
    TD(0) algorithm for value function estimation with reward tracking and heatmap visualization.
    """
    V = defaultdict(float)
    env.seed(random.randint(0, num_episodes))

    rewards_per_episode = []  # To track rewards per episode
    # To store value function for heatmaps every `heatmap_interval` episodes
    heatmap_snapshots = {}

    for episode in range(1, num_episodes + 1):
        state = deepcopy(env.reset())
        done = False
        episode_reward = 0  # Track cumulative reward for the episode

        while not done:
            agent_deck = state['agent_deck']
            state_tuple = (tuple(state['agent_deck']),
                           tuple(state['opponent_deck']),
                           state['rps_res'])

            action = random.randint(0, len(agent_deck) - 1)
            new_state, reward, done = env.step(action)
            episode_reward += reward  # Accumulate reward for the episode

            new_state_tuple = (tuple(new_state['agent_deck']),
                               tuple(new_state['opponent_deck']),
                               new_state['rps_res'])

            V[state_tuple] += alpha * \
                (reward + gamma * V.get(new_state_tuple, 0) - V.get(state_tuple, 0))
            state = deepcopy(new_state)
        
        if episode % 100 == 0:    
            rewards_per_episode.append(episode_reward)

        # Capture value function snapshots for heatmap every `heatmap_interval` episodes
        if episode % heatmap_interval == 0:
            heatmap_snapshots[episode] = deepcopy(V)

    return V, rewards_per_episode, heatmap_snapshots


def plot_rewards(rewards_per_episode):
    """
    Plot the reward per episode to visualize the convergence of the algorithm.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards_per_episode)), rewards_per_episode,
             label="Reward per Episode", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward per Episode over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_heatmap(V, title):
    """
    Plot a heatmap to visualize the value function.
    """
    # Convert the value function (V) into a matrix-like structure
    states = list(V.keys())
    agent_decks = list([s[0] for s in states])
    opponent_decks = list([s[1] for s in states])

    heatmap_matrix = []
    for agent_deck in agent_decks:
        row = []
        for opponent_deck in opponent_decks:
            state1 = (agent_deck, opponent_deck, 0)
            state2 = (agent_deck, opponent_deck, 1)
            max_v = max(V.get(state1,0), V.get(state2,0))
            # Default to 0 if the state is not in V
            row.append(max_v)
        heatmap_matrix.append(row)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_matrix, annot=False, cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.xlabel("Opponent Decks")
    plt.ylabel("Agent Decks")
    plt.show()


def td_zero_main(n=8, num_episodes=10000, alpha=0.1, gamma=1.0, heatmap_interval=2000):
    env = CardGameEnv(n)

    # Run TD(0) and get the value function, rewards, and heatmap snapshots
    V, rewards_per_episode, heatmap_snapshots = td_zero(
        env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, heatmap_interval=heatmap_interval
    )

    print("Final Estimated Value Function:")
    for state, value in V.items():
        print(f"State: {state}, Value: {value}")

    # Plot reward per episode to show convergence
    plot_rewards(rewards_per_episode)

    # Generate heatmaps every `heatmap_interval` episodes
    for episode, V_snapshot in heatmap_snapshots.items():
        plot_heatmap(V_snapshot, title=f"Value Function Heatmap at {episode} Episodes")


if __name__ == "__main__":
    td_zero_main()
