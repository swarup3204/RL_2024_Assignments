# Card Playing Meets Rock-Paper-Scissors

## Overview

This project involves implementing reinforcement learning algorithms and simulating a card game environment. The primary components include Value Iteration (VI), Temporal Difference Learning (TD), game simulation, and the card game environment.

## Table of Contents

- [Value Iteration (VI)](#value-iteration-vi)
- [Temporal Difference Learning (TD)](#temporal-difference-learning-td)
- [Simulate Game](#simulate-game)
- [Card Game Environment](#card-game-environment)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Value Iteration (VI)

Value Iteration is an algorithm used to compute the optimal policy in a Markov Decision Process (MDP). It iteratively updates the value function until convergence.

### Code

The code for Value Iteration is located in `value_iteration.py`. This file contains the implementation of the VI algorithm, and also prints the state values, optimal policies and shows convergence of value function on 100 random states.

#### Key Functions

- `value_iteration()`: Performs the value iteration algorithm to compute the optimal value function and policy.

## Temporal Difference Learning (TD)

Temporal Difference Learning is a reinforcement learning technique used to estimate the value of states and improve policies based on observed rewards.

### Code

The TD learning implementation can be found in `temporal_difference.py`. This file includes methods for updating state values and policies based on TD learning. This also prints a reward convergence and value convergence line graph for randomly selected states.

#### Key Functions

- `td_zero()`: Updates the value function using temporal difference methods.

## Simulate Game

The game simulation code is responsible for running the card game, collecting rewards, and transitioning between states.

### Code

The simulation code is located in `simulate_game.py`. This file handles the interaction between the agent and the environment during gameplay.

#### Key Functions

- `simulate_game()`: Runs a simulation of the card game, interacting with the environment and collecting rewards.

## Card Game Environment

The card game environment provides the rules and mechanics of the card game. It defines how the game is played and how states transition based on actions.

### Code

The environment code is in `CardGameEnv.py`. This file defines the environment's state, action space, and reward structure.

#### Key Classes

- `CardGameEnvironment`: The class representing the card game environment, including methods for state transitions and reward calculations.

## Requirements

To run the code, ensure you have the following dependencies installed:

- Python 3.x
- `numpy`
- `deepcopy` (if not available in your Python distribution)
- Other dependencies as specified in `requirements.txt`

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

You are free to tamper with the hyperparameters in each of these files. The defaults are explained in report.
1. **Value Iteration**: Run `python3 value_iteration.py` to compute the optimal policy using value iteration.
2. **Temporal Difference Learning**: Execute `python3 temporal_difference.py` to apply TD learning and train value functions.
3. **Simulate Game**: Use `python3 simulate_game.py` to run simulations of the card game.
4. **Card Game Environment**: Initialize and interact with the environment using `card_game_environment.py`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.
<!-- 
## License

This project is licensed under the [MIT License](LICENSE).

--- -->

Feel free to modify or expand upon this template based on your project's specifics and requirements!
