import numpy as np
import random

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.9995
num_episodes = 10000

# State space (discretized)
vertical_diff_bins = np.linspace(-200, 200, 20)
velocity_bins = np.linspace(-10, 10, 10)
horizontal_distance_bins = np.linspace(0, 300, 10)

# Q-table: [vertical_diff, velocity, horizontal_distance, action]
q_table = np.zeros((21, 11, 11, 2))  # +1 for out-of-range values

# Function to get the current state as bin indices
def get_state(bird_y, pipe_gap_y, velocity, pipe_x, bird_x):
    vertical_diff = bird_y - pipe_gap_y
    horizontal_distance = pipe_x - bird_x
    vd_bin = np.digitize(vertical_diff, vertical_diff_bins)
    vel_bin = np.digitize(velocity, velocity_bins)
    hd_bin = np.digitize(horizontal_distance, horizontal_distance_bins)
    return vd_bin, vel_bin, hd_bin

# Training loop
for episode in range(num_episodes):
    game_over = False
    score = 0
    state = get_initial_state()  # Assume this resets the game and returns initial state

    while not game_over:
        if random.random() < epsilon:
            action = random.randint(0, 1)  # Explore: random action
        else:
            action = np.argmax(q_table[state])  # Exploit: best action from Q-table

        # Take action and get next state, reward
        next_state, reward, game_over = take_action(action)  # Assume this simulates the action

        # Update Q-table
        if game_over:
            target = reward
        else:
            target = reward + gamma * np.max(q_table[next_state])
        q_table[state][action] += alpha * (target - q_table[state][action])

        state = next_state
        score += 1

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Optional: Print progress
    if episode % 100 == 0:
        print(f"Episode {episode}, Score: {score}, Epsilon: {epsilon:.3f}")

# Testing the AI
epsilon = 0  # No exploration
game_over = False
state = get_initial_state()

while not game_over:
    action = np.argmax(q_table[state])
    state, reward, game_over = take_action(action)
    # Render the game if desired