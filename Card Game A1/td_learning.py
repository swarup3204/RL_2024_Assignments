import random
from collections import defaultdict
from copy import deepcopy  # Import deepcopy
from CardGameEnv import CardGameEnv


def td_zero(env, num_episodes=1000, alpha=0.1, gamma=1, epsilon=0.1):
    """
    TD(0) Algorithm for estimating the value function.
    
    Parameters:
    - env: The environment instance of the card game.
    - num_episodes: Number of episodes to run for training.
    - alpha: Learning rate.
    - gamma: Discount factor for future rewards.
    - epsilon: Probability for epsilon-greedy action selection.
    
    Returns:
    - V: The estimated value function.
    """
    # Initialize the value function
    V = defaultdict(float)  # Default value is 0 for all states

    for episode in range(num_episodes):
        # Initialize the environment and get the initial state
        state = deepcopy(env.reset(seed = random.randint(0, 100000000)))  # Use deepcopy here
        done = False

        while not done:
            agent_deck = state['agent_deck']
            # print(state)

            # Epsilon-greedy action selection
            action = random.randint(0, len(agent_deck) - 1)  # Random action

            # Take the action and observe the new state and reward
            new_state_c, reward, done = env.step(action)
            new_state = deepcopy(new_state_c)  # Use deepcopy here
            new_state_tuple = (tuple(new_state['agent_deck']),
                               tuple(new_state['opponent_deck']),
                               new_state['opponent_card_shown'])

            state_tuple = (tuple(state['agent_deck']),
                           tuple(state['opponent_deck']),
                           state['opponent_card_shown'])

            # print(state, new_state)
            # either terminal state or doesn't exist in dictionary
            if done:
                V[new_state_tuple] = 0
                # print(new_state_tuple)

            # Update the value function using the TD(0) update rule
            V[state_tuple] += alpha * \
                (reward + gamma * (V.get(new_state_tuple,0) - V.get(state_tuple,0)))

            # Move to the next state
            state = deepcopy(new_state)  # Use deepcopy here
        # print(len(V))

    return V


def main():
    env = CardGameEnv(n=8)

    # Train the value function using TD(0)
    value_function = td_zero(env, num_episodes=1000000,
                             alpha=0.1, gamma=1, epsilon=0.1)
    print(len(value_function))
    # Print the estimated value function
    print("Estimated Value Function:")
    for state, value in value_function.items():
        print(f"State: {state}, Value: {value}")


if __name__ == "__main__":
    main()
