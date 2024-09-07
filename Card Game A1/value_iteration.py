import itertools
from collections import defaultdict
from copy import deepcopy
from CardGameEnv import CardGameEnv


def generate_all_states(n):
    """
    Generate all possible states for the agent's and opponent's decks.
    """
    cards = list(range(1, n + 1))
    states = []

    # Generate all permutations for decks
    for k in range(0, n // 2 + 1):
        agent_permutations = list(itertools.combinations(cards, k))
        for agent_deck in agent_permutations:
            remaining_cards = set(cards) - set(agent_deck)
            opponent_permutations = list(
                itertools.permutations(remaining_cards, n // 2))
            for opponent_deck in opponent_permutations:
                states.append((agent_deck, opponent_deck, 0))
                states.append((agent_deck, opponent_deck, 1))
    return states


def value_iteration(env, gamma=1.0, theta=1e-7):
    """
    Value Iteration Algorithm for policy improvement.
    """
    all_states = generate_all_states(env.n)
    print("Total number of states in VI:", len(all_states))
    V = defaultdict(float)
    policy = {}

    while True:
        delta = 0
        for state in all_states:
            agent_deck, opponent_deck, rps_res = state

            if len(agent_deck) == 0:
                V[state] = 0
                continue

            v = V[state]
            best_value = float('-inf')

            for action in range(len(agent_deck)):
                env.state = deepcopy({
                    'agent_deck': list(agent_deck),
                    'opponent_deck': list(opponent_deck),
                    'rps_res': rps_res
                })

                new_state, reward, done = env.step(action)
                new_state_tuple = (tuple(new_state['agent_deck']),
                                   tuple(new_state['opponent_deck']),
                                   new_state['rps_res'])

                action_value = reward + gamma * V.get(new_state_tuple, 0)

                if action_value > best_value:
                    best_value = action_value
                    policy[state] = action

                if done:
                    break

            V[state] = best_value
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    return V, policy


def value_iteration_main(n=8, gamma=1.0, theta=1e-7):
    env = CardGameEnv(n)
    V, policy = value_iteration(env, gamma=gamma, theta=theta)

    print("Optimal Value Function:")
    for state, value in V.items():
        print(f"State: {state}, Value: {value}")

    print("\nOptimal Policy:")
    for state, action in policy.items():
        print(f"State: {state}, Action: {action}")


if __name__ == "__main__":
    value_iteration_main()
