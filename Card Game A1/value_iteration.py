#Roll: 20CS30062
#CAV1

import random
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations, permutations
from copy import deepcopy
from CardGameEnv import CardGameEnv


def generate_all_states(n):
    """
    Generate all possible states with agent deck and opponent_card_shown.
    """
    cards = list(range(1, n + 1))
    states = []

    for k in range(0, n // 2 + 1):
        agent_permutations = list(combinations(cards, k))
        for agent_deck in agent_permutations:
            remaining_cards = set(cards) - set(agent_deck)
            # Generate all possible opponent decks from remaining cards
            opponent_deck_permutations = list(
                permutations(remaining_cards, k))
            for opponent_deck in opponent_deck_permutations:
                states.append((agent_deck, opponent_deck))
    return states


def value_iteration(env, gamma=0.9, theta=1e-4, n_iterations=500, n_states_to_track=100):
    """
    Value Iteration Algorithm for policy improvement with updated state variables.
    """
    all_states = generate_all_states(env.n)
    print("Number of states: ", len(all_states))

    V = defaultdict(float)  # Default value initialization to 0
    policy = {}
    cnt = 0
    random.seed(42)

    tracked_states = []  # List to store 100 randomly selected state tuples
    tracked_values = {}  # Dictionary to store values for tracked states

    while True:
        cnt += 1
        delta = 0  # Keep track of maximum change
        vis = defaultdict(bool)

        for state in all_states:
            agent_deck, opponent_deck = state
            agent_deck = list(agent_deck)
            opponent_deck = list(opponent_deck)
            state_tuple = (tuple(agent_deck), opponent_deck[0] if len(
                opponent_deck) > 0 else -1)

            if len(agent_deck) == 0:
                V[state_tuple] = 0  # If no cards left, the value is 0
                continue

            v = V[state_tuple]
            if vis.get(state_tuple, False):
                best_value = float('-inf')
            else:
                best_value = v

            for action in range(len(agent_deck)):
                # Initialize the environment state
                env.state = deepcopy({
                    'agent_deck': agent_deck,
                    'opponent_card_shown': opponent_deck[0]
                })
                env.opponent_deck = deepcopy(opponent_deck)

                # Take a step and get the next state, reward, and done flag
                new_state, reward, done = env.step(action)
                new_state_tuple = (
                    tuple(new_state['agent_deck']),
                    new_state['opponent_card_shown']
                )

                action_value = reward + gamma * V.get(new_state_tuple, 0)

                if action_value > best_value:
                    best_value = action_value
                    policy[state_tuple] = action

            V[state_tuple] = best_value
            # print(state_tuple)
            delta = max(delta, abs(v - V[state_tuple]))

            # Randomly select 100 states for tracking after state_tuple is created
            if len(tracked_states) < n_states_to_track and state_tuple not in tracked_states and len(list(state_tuple[0])) > 3:
                tracked_states.append(state_tuple)
                tracked_values[state_tuple] = []

            # Store the value of the tracked states at this iteration
            if state_tuple in tracked_states:
                tracked_values[state_tuple].append(V[state_tuple])

        print("Delta: ", delta)
        # this limit cnt > n_iterations is not part of algorithm and is never reached, however it is added to avoid infinite loop
        if delta < theta or cnt > n_iterations:
            print("Iterations: ", cnt, "Delta: ", delta)
            break

    return V, policy, tracked_values


def plot_convergence(tracked_values):
    
    """
    Plot the convergence of value function for 100 tracked states.
    """
    plt.figure(figsize=(10, 6))
    for state, values in tracked_values.items():
        plt.plot(values, label=f'State: {state}', alpha=0.8)
        print(f"State: {state}, Values: {values}")

    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title(
        'Convergence of Value Function over Iterations for 100 Randomly Selected States')
    plt.grid(True)
    plt.show()


def value_iteration_main(n=10, gamma=0.8, theta=1e-1):
    env = CardGameEnv(n)
    V, policy, tracked_values = value_iteration(
        env, gamma=gamma, theta=theta, n_iterations=500, n_states_to_track=10)

    print("Optimal Value Function:")
    for state, value in V.items():
        print(f"State: {state}, Value: {value}")

    print("\nOptimal Policy:")
    for state, action in policy.items():
        print(f"State: {state}, Action: {action}")

    # Plot the convergence of value function
    plot_convergence(tracked_values)


if __name__ == "__main__":
    value_iteration_main()


