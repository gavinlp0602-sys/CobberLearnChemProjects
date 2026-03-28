import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create the FrozenLake environment
# 'is_slippery=False' makes it deterministic (easier for debugging)
env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)

# Reset the environment to get the initial state
observation, info = env.reset()

# 2. Get the dimensions of our state and action spaces
state_size = env.observation_space.n
action_size = env.action_space.n

# 3. Create the Q-Table initialized with zeros
# Rows = States (16), Columns = Actions (4)
q_table = np.zeros((state_size, action_size))

print("Initial Q-Table (First 5 states):")
print(q_table[:5])
print(f"\nQ-Table Shape: {q_table.shape}")

# --- Hyperparameters ---
episodes = 200          # Increased to give more time to explore
learning_rate = 0.05      # Lowered slightly for more stable updates
discount_factor = 0.95    # Standard for FrozenLake
epsilon = 1.0
epsilon_decay = 0.0002    # Slower decay to keep exploring longer
min_epsilon = 0.01

# Tracking rewards
rewards_per_episode = []

for i in range(episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not (terminated or truncated):
        # 1. Choose Action (Epsilon-Greedy Policy)
        if np.random.random() < epsilon:
            action = env.action_space.sample() # Explore
        else:
            action = np.argmax(q_table[state, :]) # Exploit

        # 2. Take Action
        next_state, reward, terminated, truncated, info = env.step(action)

        # 3. Update Q-Table (Bellman Equation)
        # New Q = Old Q + LR * [Reward + (Gamma * Max Next Q) - Old Q]
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon - epsilon_decay)
    rewards_per_episode.append(total_reward)

# 4. Plotting the Success Rate
# We calculate the average reward for every 10 episodes
chunk_size = 10
avg_rewards = [np.mean(rewards_per_episode[i:i+chunk_size])
               for i in range(0, episodes, chunk_size)]

plt.plot(range(0, episodes, chunk_size), avg_rewards)
plt.title('Agent Success Rate (FrozenLake-v1)')
plt.xlabel('Episodes')
plt.ylabel('Average Reward (Success Rate)')
plt.grid(True)
plt.show()

print("Training complete. The agent has filled the Q-Table!")

# Close the environment
env.close()


# 1. Save and Load Functions
def save_q_table(table, filename="frozen_lake_q_table.npy"):
    np.save(filename, table)
    print(f"Q-Table saved to {filename}")


def load_q_table(filename="frozen_lake_q_table.npy"):
    return np.load(filename)


# 2. Interpretation Function
def interpret_q_table(table):
    """Prints the Q-Table and the best action for each state."""
    # Mapping action indices to cardinal directions
    # 0: Left, 1: Down, 2: Right, 3: Up
    actions = ["Left ", "Down ", "Right", "Up   "]

    print("\n--- Q-Table Interpretation ---")
    print("State |  Left  |  Down  |  Right |   Up   | Best Move")
    print("-" * 55)

    for state_idx, row in enumerate(table):
        # Find the index of the highest value in this state's row
        best_action_idx = np.argmax(row)
        best_move = actions[best_action_idx]

        # Check if the state has been 'learned' (not all zeros)
        if np.max(row) > 0:
            row_str = " | ".join([f"{val:.3f}" for val in row])
            print(f" {state_idx:02d}   | {row_str} | {best_move}")
        else:
            print(f" {state_idx:02d}   | 0.000 | 0.000 | 0.000 | 0.000 | (Unknown)")


# --- Usage Example ---
# Assuming 'q_table' was created in the previous step:
save_q_table(q_table)
interpret_q_table(q_table)