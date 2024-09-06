import itertools
from collections import defaultdict
from copy import deepcopy
from CardGameEnv import CardGameEnv


def generate_all_states(n):
    """
    Generate all possible states based on the agent's deck, opponent's deck, and opponent's card shown.
    This function generates states using permutations instead of combinations, as the order of cards matters.
    
    Parameters:
    - n: Total number of cards.
    
    Returns:
    - states: A list of all possible states.
    """
    cards = list(range(1, n + 1))
    states = []

    # Generate all possible sizes for the agent's deck (from 0 to n//2 cards)
    for k in range(0, n // 2 + 1):
        # Generate all permutations of size k for the agent's deck
        agent_deck_permutations = list(itertools.permutations(cards, k))

        for agent_deck in agent_deck_permutations:
            # Opponent deck is the remaining cards (considering permutations as well)
            remaining_cards = set(cards) - set(agent_deck)
            opponent_deck_permutations = list(
                itertools.permutations(remaining_cards, k))

            for opponent_deck in opponent_deck_permutations:
                # Include all possible states of opponent card shown (-1 means no card shown)
                for opponent_card_shown in (-1, *opponent_deck):
                    states.append(
                        (agent_deck, opponent_deck, opponent_card_shown))

    return states


def value_iteration(env, gamma=1, theta=1e-5):
    """
    Value Iteration Algorithm for Policy Improvement.
    
    Parameters:
    - env: The environment instance of the card game.
    - gamma: Discount factor for future rewards.
    - theta: Threshold for the value function convergence.
    
    Returns:
    - V: Optimal value function.
    - policy: Optimal policy derived from the value function.
    """
    # Generate all possible states
    all_states = generate_all_states(env.n)
    print(f"Total states generated: {len(all_states)}")

    # Initialize value function and policy dictionaries
    V = defaultdict(float)  # Default value is 0 for all states
    policy = {}

    while True:
        delta = 0  # Initialize the change in value function to 0

        # Iterate over all states
        for state in all_states:
            agent_deck, opponent_deck, opponent_card_shown = state

            # Check if this is a terminal state
            if len(agent_deck) == 0 or len(opponent_deck) == 0:
                V[state] = 0  # Value of terminal state is 0
                continue  # No need to process terminal states

            v = V[state]  # Store the current value for this state
            best_value = float('-inf')  # Initialize the best value

            # For each card in the agent's deck, evaluate the possible action
            for action in range(len(agent_deck)):
                # Create a new environment state
                env.state = deepcopy({
                    'agent_deck': list(agent_deck),
                    'opponent_deck': list(opponent_deck),
                    'opponent_card_shown': opponent_card_shown
                })

                # Take the action and observe the resulting state and reward
                new_state, reward, done = env.step(action)

                # Create a tuple for the new state
                new_state_tuple = (tuple(new_state['agent_deck']),
                                   tuple(new_state['opponent_deck']),
                                   new_state['opponent_card_shown'])

                # Calculate the value of this action
                action_value = reward + gamma * V.get(new_state_tuple, 0)

                # Update the best value and policy
                if action_value > best_value:
                    best_value = action_value
                    policy[state] = action

                if done:
                    break

            # Update the value function with the best value found
            V[state] = best_value

            # Update the change in value function
            delta = max(delta, abs(v - V[state]))

        # Check for convergence
        if delta < theta:
            break

    return V, policy


def main():
    env = CardGameEnv(n=8)

    # Get the optimal value function and policy
    optimal_value_function, optimal_policy = value_iteration(env)

    # Print the optimal value function and policy
    print("Optimal Value Function:")
    for state, value in optimal_value_function.items():
        print(f"State: {state}, Value: {value}")

    print("\nOptimal Policy:")
    for state, action in optimal_policy.items():
        print(f"State: {state}, Action: {action}")


if __name__ == "__main__":
    main()
