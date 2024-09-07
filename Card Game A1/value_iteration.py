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


def value_iteration(env, gamma=0.9, theta=1e-4):
    """
    Value Iteration Algorithm for policy improvement with updated state variables.
    """
    all_states = generate_all_states(env.n)
    print("Number of states: ", len(all_states))
    # print(all_states)
    # return {}, {}

    V = defaultdict(float)  # Default value initialization to 0
    policy = {}
    cnt = 0
    
    while True:
        cnt+=1
        delta = 0  # Keep track of maximum change
        changer = None
        vis = defaultdict(bool)

        for state in all_states:
            agent_deck, opponent_deck = state
            agent_deck = list(agent_deck)
            opponent_deck = list(opponent_deck)
            state_tuple = (tuple(agent_deck), opponent_deck[0] if len(opponent_deck) > 0 else -1)

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
                    changer = new_state_tuple

                # print("Agent deck: ", agent_deck, "Opponent deck: ", opponent_deck)
                # Debugging: Print step details
                # print(f"State: {state}, Action: {action}, Reward: {reward}, "
                #       f"New State: {new_state_tuple}, Value: {action_value}")

            # print(f"Scenario: {state}, State: {state_tuple}, Changer: {changer}, New v: {best_value}")
            V[state_tuple] = best_value
            # if abs(v - V[state_tuple]) > delta:
            #     print("Here exceeded", "Diff:", abs(v - V[state_tuple]), "Scenario:", state, "State:", state_tuple, "Original v:", v, "Modified", V[state_tuple],"Cnt",cnt, "Changer:", changer)
            delta = max(delta, abs(v - V[state_tuple]))

        
        print("Delta: ", delta)
        if delta < theta or cnt > 400:
            print("Cnt: ", cnt, "Delta: ", delta)
            break
        
        # return V, policy
        
    return V, policy


def value_iteration_main(n=10, gamma=0.8, theta=1e-1):
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
