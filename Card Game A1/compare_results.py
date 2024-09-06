import matplotlib.pyplot as plt
from value_iteration import value_iteration
from td_learning import td_zero
from CardGameEnv import CardGameEnv

# Initialize the environment
env = CardGameEnv(n=8)

# Get value functions from Value Iteration and TD Learning
# Assuming value_iteration returns (V, policy)
vi_values_dict, _ = value_iteration(env)
td_values_dict = td_zero(env)  # Assuming td_zero returns V
print(len(td_values_dict))

# Create a unique index for each state (since states are tuples)
# Sort the states (tuples) for consistent plotting
states = sorted(vi_values_dict.keys())
state_indices = range(len(states))  # Assign a unique index for each state

# Extract the values from the defaultdict for Value Iteration and TD Learning
vi_values = [vi_values_dict[state] if isinstance(
    vi_values_dict[state], (int, float)) else vi_values_dict[state][0] for state in states]
td_values = [td_values_dict[state] if state in td_values_dict and isinstance(
    td_values_dict[state], (int, float)) else 0 for state in states]

# Check for any nested lists or non-scalar values (can modify based on your needs)


def flatten_values(value):
    if isinstance(value, list) or isinstance(value, tuple):
        # Assuming you want to extract the first element if it's a list/tuple
        return value[0]
    return value


# Apply the flattening function
vi_values = [flatten_values(v) for v in vi_values]
td_values = [flatten_values(v) for v in td_values]

# Plot the results for comparison
plt.plot(state_indices, vi_values, label="Value Iteration", marker='o')
plt.plot(state_indices, td_values, label="TD Learning", marker='x')
plt.xlabel("State Index")
plt.ylabel("State Values")
plt.legend()
plt.title("Comparison of Value Iteration and TD Learning")
plt.show()
