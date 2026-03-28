import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# --- 1. Hyperparameters ---
BATCH_SIZE = 64
LR = 0.001  # Learning Rate
GAMMA = 0.99  # Discount Factor (how much we care about future rewards)
EPS_START = 1.0  # Starting Exploration (100% random actions)
EPS_END = 0.05  # Minimum Exploration
EPS_DECAY = 200  # How quickly epsilon decays (number of episodes)
MEMORY_SIZE = 10000  # Size of Replay Buffer
TARGET_UPDATE = 10  # How often to update the target network


# --- 2. The Neural Network ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Input layer (4 obs) -> Hidden Layer (128 neurons)
        self.layer1 = nn.Linear(n_observations, 128)
        # Hidden Layer -> Output layer (2 actions)
        self.layer2 = nn.Linear(128, n_actions)

    def forward(self, x):
        # Activation function (ReLU) introduces non-linearity
        x = torch.relu(self.layer1(x))
        return self.layer2(x)  # Returns Q-values for each action


# --- 3. The Replay Memory ---
# Stores experienced transitions so we can learn from them later (breaking correlation)
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- 4. The Agent ---
class Agent:
    def __init__(self, n_observations, n_actions):
        self.n_actions = n_actions
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Match initially
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, episode):
        # Epsilon-greedy strategy: explore vs exploit
        epsilon = EPS_END + (EPS_START - EPS_END) * \
                  np.exp(-1. * episode / EPS_DECAY)

        self.steps_done += 1
        if random.random() > epsilon:
            # Exploit: Select the action with the highest Q-value
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Explore: Select a random action
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # 1. Sample a batch
        transitions = self.memory.sample(BATCH_SIZE)
        # Batch-transpose (unpack the tuples into separate tensors)
        batch = list(zip(*transitions))

        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        next_state_batch = torch.cat(batch[2])
        reward_batch = torch.cat(batch[3])
        done_batch = torch.cat(batch[4])

        # 2. Get current Q-values from the Policy Net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 3. Get expected Q-values from Target Net (Bellman Equation)
        # We don't track gradients for the target net
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]

        # If the episode ended (done=True), expected_q is just the reward.
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

        # 4. Compute Loss (Huber Loss is robust to outliers)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 5. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --- 5. Main Training Loop ---
env = gym.make("CartPole-v1", max_episode_steps=500)
# Extract observation/action space details
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = Agent(n_observations, n_actions)
episode_durations = []

NUM_EPISODES = 250

print("Starting training... This may take a few minutes.")
for episode in range(NUM_EPISODES):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    current_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, episode)
        next_state_raw, reward, terminated, truncated, _ = env.step(action.item())
        current_reward += reward
        done = terminated or truncated

        # Convert everything to tensors for PyTorch
        reward = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([float(done)], dtype=torch.float32)
        next_state = torch.tensor(next_state_raw, dtype=torch.float32).unsqueeze(0)

        # Store transition in memory
        agent.memory.push(state, action, next_state, reward, done_tensor)

        state = next_state

        # Perform one step of optimization
        agent.optimize_model()

    episode_durations.append(current_reward)

    # Update the Target Network periodically
    if episode % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    if episode % 25 == 0:
        avg_reward = np.mean(episode_durations[-25:]) if len(episode_durations) > 25 else np.mean(episode_durations)
        print(f"Episode: {episode} | Last 25 Avg Reward: {avg_reward:.1f}")

env.close()

# --- 6. Plotting Results ---
plt.figure(figsize=(10, 5))
plt.title('DQN Learning Curve (CartPole-v1)')
plt.xlabel('Episode')
plt.ylabel('Total Reward (Duration)')
plt.plot(episode_durations, label='Episode Reward', alpha=0.3, color='blue')

# Plot a moving average to smooth the noise
if len(episode_durations) >= 50:
    means = np.convolve(episode_durations, np.ones(50) / 50, mode='valid')
    means = np.concatenate((np.zeros(49), means))  # Align lengths
    plt.plot(means, label='50-Episode Moving Avg', color='red', linewidth=2)

plt.axhline(y=500, color='green', linestyle='--', label='Perfect Score')
plt.legend()
plt.grid(True)
plt.show()