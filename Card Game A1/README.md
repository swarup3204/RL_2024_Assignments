# Card Playing meets Rock-Paper-Scissors

## Overview

This project implements an RL environment for a card game where players first play a game of rock-paper-scissors, and then the winner selects a card. The agent's goal is to maximize the number of rounds won using RL algorithms: Value Iteration (VI) and Temporal Difference (TD) Learning.

## Files

- `CardGameEnv.py`: Contains the environment class for the game.
- `value_iteration.py`: Implements Value Iteration algorithm.
- `td_learning.py`: Implements Temporal Difference Learning algorithm.
- `compare_results.py`: Compares the results of VI and TD and plots them.
- `README.md`: This file.

## How to Run

1. Ensure Python 3.x is installed.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run `python compare_results.py` to see the comparison between VI and TD Learning.

## Results

The results show the value functions learned by VI and TD Learning, which can be compared visually using the plot generated in `compare_results.py`.
