import gym
from gym import spaces
import random


# class CardGameEnv(gym.Env):
#     """
#     Custom Environment for the Card Playing meets Rock-Paper-Scissors game.
#     """

#     def __init__(self, n=50):
#         super(CardGameEnv, self).__init__()

#         # Number of cards in the deck
#         self.n = n
#         self.state = {"agent_deck": [],
#                       "opponent_deck": [], "opponent_card_shown": -1}

#         self.current_step = 0

#     def _determine_opponent_card_shown(self):
#         """
#         Determine if the opponent won RPS and show the top card of the opponent's deck if true.
#         Otherwise, return -1 (indicating no card is shown).
#         """
#         if random.choice([True, False]) and len(self.state['opponent_deck']) > 0:  # If opponent wins RPS
#             # Show the top card of the opponent's deck
#             return self.state['opponent_deck'][0]
#         return -1  # No card is shown

#     def step(self, action):
#         """
#         Take an action (agent selects a card) and return the results of the round.
#         """
#         agent_deck = self.state['agent_deck']
#         opponent_deck = self.state['opponent_deck']

#         # Select the agent's card based on the action
#         agent_card = agent_deck[action]

#         # Remove the selected card from the agent's deck
#         agent_deck.pop(action)

#         # Select the opponent's top card
#         opponent_card = opponent_deck[0]
#         opponent_deck.pop(0)
#         # print("Agent Card:", agent_card, "Opponent Card:", opponent_card)
        
#         # Calculate the reward
#         reward = 1 if agent_card > opponent_card else -1

#         # Update the state with the current information
#         self.state = {
#             'agent_deck': agent_deck,
#             'opponent_deck': opponent_deck
#         }
        
#         # print(self.state['opponent_deck'])
#         self.state['opponent_card_shown'] = self._determine_opponent_card_shown()

#         # Increment the step counter
#         self.current_step += 1

#         # Check if the game is done (all cards have been played)
#         done = (len(opponent_deck) == 0)

#         return self.state, reward, done

#     def reset(self):
#         """
#         Reset the environment to the initial state.
#         """
#         # Create new shuffled decks for both agent and opponent
#         all_cards = list(range(1, self.n + 1))
#         random.shuffle(all_cards)

#         self.opponent_deck = all_cards[:self.n // 2]
#         self.agent_deck = all_cards[self.n // 2:]

#         # Determine if the opponent won RPS and show the top card if true
#         self.opponent_card_shown = self._determine_opponent_card_shown()

#         # Reset the state and step counter
#         self.current_step = 0

#         self.state = {
#             'agent_deck': self.agent_deck,
#             'opponent_deck': self.opponent_deck,
#             'opponent_card_shown': self.opponent_card_shown
#         }

#         return self.state

#     def render(self):
#         """
#         Print the current state of the game.
#         """
#         print(f"Agent Deck: {self.state['agent_deck']}, Opponent Deck: {self.state['opponent_deck']}, Opponent Card Shown: {self.state['opponent_card_shown']}")


class CardGameEnv:
    """
    Custom Environment for the Card Playing meets Rock-Paper-Scissors game.
    """

    def __init__(self, n=10):
        # Number of cards in the deck
        super(CardGameEnv, self).__init__()
        self.n = n
        self.state = {"agent_deck": [],
                      "opponent_deck": [], "opponent_card_shown": -1}
        self.current_step = 0
        # self.seed_val = 32
        # self.seed(self.seed_val)

    # def seed(self, seed=None):
    #     """
    #     Set the seed for reproducibility.
    #     """
    #     self.seed_val = seed
    #     random.seed(self.seed_val)

    def _determine_opponent_card_shown(self):
        """
        Determine if the opponent won RPS and show the top card of the opponent's deck if true.
        Otherwise, return -1 (indicating no card is shown).
        """
        if random.choice([True, False]) and len(self.state['opponent_deck']) > 0:  # If opponent wins RPS
            # Show the top card of the opponent's deck
            return self.state['opponent_deck'][0]
        return -1  # No card is shown

    def step(self, action):
        """
        Take an action (agent selects a card) and return the results of the round.
        """
        agent_deck = self.state['agent_deck']
        opponent_deck = self.state['opponent_deck']

        # Select the agent's card based on the action
        agent_card = agent_deck[action]

        # Remove the selected card from the agent's deck
        agent_deck.pop(action)

        # Select the opponent's top card
        opponent_card = opponent_deck[0]
        opponent_deck.pop(0)

        # Calculate the reward
        reward = 1 if agent_card > opponent_card else -1

        # Update the state with the current information
        self.state = {
            'agent_deck': agent_deck,
            'opponent_deck': opponent_deck
        }

        # Update opponent's card shown based on RPS
        self.state['opponent_card_shown'] = self._determine_opponent_card_shown()

        # Increment the step counter
        self.current_step += 1

        # Check if the game is done (all cards have been played)
        done = (len(opponent_deck) == 0)

        return self.state, reward, done

    def reset(self, seed = None):
        """
        Reset the environment to the initial state.
        """
        # Create new shuffled decks for both agent and opponent
        random.seed(seed)

        all_cards = list(range(1, self.n + 1))
        random.shuffle(all_cards)

        self.opponent_deck = all_cards[:self.n // 2]
        self.agent_deck = all_cards[self.n // 2:]

        # Determine if the opponent won RPS and show the top card if true
        self.opponent_card_shown = self._determine_opponent_card_shown()

        # Reset the state and step counter
        self.current_step = 0

        self.state = {
            'agent_deck': self.agent_deck,
            'opponent_deck': self.opponent_deck,
            'opponent_card_shown': self.opponent_card_shown
        }

        return self.state

    def render(self):
        """
        Print the current state of the game.
        """
        print(f"Agent Deck: {self.state['agent_deck']}, Opponent Deck: {
              self.state['opponent_deck']}, Opponent Card Shown: {self.state['opponent_card_shown']}")
