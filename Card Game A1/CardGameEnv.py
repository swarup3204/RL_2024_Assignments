import random
from gym import Env


class CardGameEnv(Env):
    """
    Custom Environment for the Card Game where the opponent plays the top card from their deck.
    """

    def __init__(self, n=10, seed=None):
        """
        Initialize the card game environment.
        """
        super(CardGameEnv, self).__init__()
        self.n = n
        self.state = {"agent_deck": [], "opponent_card_shown": 0}
        self.opponent_deck = []  # Keep track of opponent's deck
        self.seed(seed)

    def seed(self, seed=None):
        """
        Set the seed for reproducibility.
        """
        self.rng = random.Random(seed)

    def step(self, action):
        """
        Agent takes an action (selects a card) and gets a reward based on comparison to opponent's card.
        """
        agent_deck = self.state['agent_deck']
        agent_card = agent_deck.pop(action)  # Select the agent's card

        opponent_card = self.opponent_deck.pop(0)  # Take the top opponent card
        reward = 1 if agent_card > opponent_card else -1  # Compare cards

        # Update the state
        self.state = {
            'agent_deck': agent_deck,
            'opponent_card_shown': self.opponent_deck[0] if len(self.opponent_deck) > 0 else -1 # Compare cards
        }

        # Game ends when opponent's deck is empty
        done = (len(self.opponent_deck) == 0)
        return self.state, reward, done

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        all_cards = list(range(1, self.n + 1))
        self.rng.shuffle(all_cards)

        self.opponent_deck = all_cards[:self.n // 2]  # Opponent deck
        self.state = {
            'agent_deck': sorted(all_cards[self.n // 2:]),  # Agent deck
            # Show first opponent card
            'opponent_card_shown': self.opponent_deck[0]
        }

        return self.state

    def render(self):
        """
        Print the current state of the game.
        """
        print(f"Agent Deck: {self.state['agent_deck']}, Opponent Deck: {self.opponent_deck}, Opponent Card Shown: {self.state['opponent_card_shown']}")
