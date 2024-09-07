import random
from gym import Env
from collections import defaultdict


class CardGameEnv(Env):
    """
    Custom Environment for the Card Game combined with Rock-Paper-Scissors.
    """

    def __init__(self, n=10, seed=None):
        """
        Initialize the card game environment.
        """
        super(CardGameEnv, self).__init__()
        self.n = n
        self.state = {"agent_deck": [], "opponent_deck": [], "rps_res": 0}
        self.seed(seed)

    def seed(self, seed=None):
        """
        Set the seed for reproducibility.
        """
        self.rng = random.Random(seed)

    def step(self, action):
        """
        Agent takes an action (selects a card) and gets a reward.
        """
        agent_deck = self.state['agent_deck']
        opponent_deck = self.state['opponent_deck']

        # Select the agent's card based on the action
        agent_card = agent_deck.pop(action)

        # Select the opponent's top card
        opponent_card = opponent_deck.pop(0)

        # Calculate the reward
        reward = 1 if agent_card > opponent_card else -1

        # Update the state
        self.state = {
            'agent_deck': agent_deck,
            'opponent_deck': opponent_deck,
            'rps_res': random.choice([0, 1])
        }

        # Game is done when all opponent cards are played
        done = (len(opponent_deck) == 0)
        return self.state, reward, done

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        all_cards = list(range(1, self.n + 1))
        self.rng.shuffle(all_cards)

        self.state = {
            'agent_deck': sorted(all_cards[self.n // 2:]),
            'opponent_deck': all_cards[:self.n // 2],
            'rps_res': self.rng.choice([0, 1])
        }
        return self.state

    def render(self):
        """
        Print the current state of the game.
        """
        print(f"Agent Deck: {self.state['agent_deck']}, Opponent Deck: {
              self.state['opponent_deck']}, RPS Result: {self.state['rps_res']}")
