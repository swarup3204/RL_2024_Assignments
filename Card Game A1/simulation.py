import random
from collections import defaultdict
from copy import deepcopy
# Assuming your environment is in a file named card_game_env.py
from CardGameEnv import CardGameEnv
# Assuming your value iteration function is in value_iteration.py
from value_iteration import value_iteration
# Assuming your TD learning function is in td_learning.py
from td_learning import td_zero


def simulate_game(env, policy, use_value_iteration=True, value_function=None):
    """
    Simulate a game using either the value iteration policy or the TD-learned value function.

    Parameters:
    - env: The environment instance.
    - policy: The policy learned through value iteration (for use_value_iteration=True).
    - use_value_iteration: Flag to decide which method to use for action selection.
    - value_function: The learned value function (for use_value_iteration=False, using TD).
    """
    state = env.reset(seed = 31)
    done = False
    total_reward = 0

    while not done:
        env.render()  # Render the state before each action
        agent_deck, opponent_deck, opponent_card_shown = tuple(
            state['agent_deck']), tuple(state['opponent_deck']), state['opponent_card_shown']
        state_tuple = (agent_deck, opponent_deck, opponent_card_shown)

        if use_value_iteration:
            # Use the policy from value iteration
            action = policy.get(
                state_tuple, random.choice(range(len(agent_deck))))
        else:
            # Use the TD-learned value function to select the best action
            best_action = None
            best_value = float('-inf')

            for action in range(len(agent_deck)):
                # Temporarily step into the new state to evaluate the value
                env_copy = deepcopy(env)
                new_state, _, _ = env_copy.step(action)
                new_state_tuple = (tuple(new_state['agent_deck']),
                                   tuple(new_state['opponent_deck']),
                                   new_state['opponent_card_shown'])

                # Get the value of the new state
                action_value = value_function.get(new_state_tuple, 0)

                if action_value > best_value:
                    best_value = action_value
                    best_action = action

            action = best_action if best_action is not None else random.choice(
                range(len(agent_deck)))

        # Print the agent's and opponent's chosen cards
        agent_card_chosen = state['agent_deck'][action] if len(
            state['agent_deck']) > action else None
        opponent_card_chosen = state['opponent_deck'][0]

        # Step in the real environment
        state, reward, done = env.step(action)
        total_reward += reward

        print(f"Agent chose card: {agent_card_chosen}")
        print(f"Opponent chose card: {opponent_card_chosen}")

        

    return total_reward


def main():
    n = 8
    env = CardGameEnv(n=n)

    # Value Iteration
    V_vi, policy_vi = value_iteration(env)
    print("Simulating game using Value Iteration policy:")
    total_reward_vi = simulate_game(env, policy_vi, use_value_iteration=True)
    print(f"Total Reward (Value Iteration): {total_reward_vi}\n")

    # TD Learning
    V_td = td_zero(env)
    print("Simulating game using TD-learned values:")
    total_reward_td = simulate_game(
        env, None, use_value_iteration=False, value_function=V_td)
    print(f"Total Reward (TD Learning): {total_reward_td}\n")


if __name__ == "__main__":
    main()
