from CardGameEnv import CardGameEnv
from copy import deepcopy
import random
from value_iteration import value_iteration
from td_learning import td_zero

def simulate_game(env, policy=None, use_value_iteration=True, value_function=None, seed=42):
    """
    Simulate a game using the value iteration policy or TD(0)-learned value function.
    """
    env.seed(seed)
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.render()
        state_tuple = (tuple(state['agent_deck']),
                       state['opponent_card_shown'])

        if use_value_iteration:
            action = policy.get(state_tuple, random.choice(
                range(len(state['agent_deck']))))
        else:
            best_action = None
            best_value = float('-inf')

            for action in range(len(state['agent_deck'])):
                env_copy = deepcopy(env)
                new_state, _, _ = env_copy.step(action)
                new_state_tuple = (
                    tuple(new_state['agent_deck']), new_state['opponent_card_shown'])
                action_value = value_function.get(
                    new_state_tuple, 0)

                if action_value > best_value:
                    best_value = action_value
                    best_action = action

            action = best_action if best_action is not None else random.choice(
                range(len(state['agent_deck'])))

        state, reward, done = env.step(action)
        if reward == 1:
            print("Agent wins round!")
        else:    
            print("Opponent wins round!")
        total_reward += reward

    if total_reward == 0:
        print("Game ends in a tie!")
    elif total_reward > 0:
        print("Agent wins the game!")
    else:
        print("Opponent wins the game!")
    return total_reward


def simulate_main(use_value_iteration=True, n=8, gamma=1.0, num_episodes=1000, alpha=0.1, seed=42):
    """
    Main function to simulate the game using either Value Iteration or TD(0).
    - use_value_iteration: If True, simulate using the Value Iteration policy. Otherwise, use TD(0).
    - n: Size of the deck.
    - gamma: Discount factor for both algorithms.
    - num_episodes: Number of episodes for TD(0) learning.
    - alpha: Learning rate for TD(0).
    - seed: Random seed for reproducibility.
    """
    env = CardGameEnv(n, seed=seed)

    if use_value_iteration:
        # Run Value Iteration and get the optimal policy
        print("Running Value Iteration...")
        V_vi, policy_vi = value_iteration(env, gamma=gamma)
        print("\nSimulating game using Value Iteration policy:")
        total_reward = simulate_game(
            env, policy=policy_vi, use_value_iteration=True, seed=seed)
        print(f"Total Reward (Value Iteration): {total_reward}")
    else:
        # Run TD(0) learning and get the estimated value function
        print("Running TD(0) Learning...")
        # print(type())
        V_td, _ = td_zero(env, num_episodes=num_episodes,
                       alpha=alpha, gamma=gamma)
        # print(type(V_td))
        print("\nSimulating game using TD(0) value function:")
        total_reward = simulate_game(
            env, use_value_iteration=False, value_function=V_td, seed=seed)
        print(f"Total Reward (TD(0)): {total_reward}")


if __name__ == "__main__":
    seed = random.randint(0, 1000)
    n = 6
    simulate_main(use_value_iteration=True, n=n, gamma=1.0,
                  num_episodes=1000, alpha=0.1, seed=seed)
    simulate_main(use_value_iteration=False, n=n, gamma=1.0,
                  num_episodes=1000, alpha=0.1, seed=seed)
