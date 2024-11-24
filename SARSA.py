import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Initialize the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)


# Set the number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n


# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 10000  # Number of episodes


# Initialize Q-table with zeros
Q = np.zeros((n_states, n_actions))


# Modified rewards for the FrozenLake environment
# Rewards: Goal +10, Near Goal +5, Other +2, Obstacles (holes) -5, Slippery areas -3
rewards = np.array([0, 2, -5, 2, 5, 2, 2, -3, 10, -5, 2, 2, 2, -5, 2, 10])


# SARSA algorithm
for episode in range(episodes):
    state = env.reset()[0]
    action = np.random.choice(n_actions) if np.random.uniform(0, 1) < epsilon else np.argmax(Q[state])


    done = False
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        reward = rewards[next_state]  # Use modified rewards


        next_action = np.random.choice(n_actions) if np.random.uniform(0, 1) < epsilon else np.argmax(Q[next_state])
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])


        state = next_state
        action = next_action


# Print the final Q-values
print("Final Q-Values:")
print(Q)


# Determine the best policy (actions) for each state based on Q-values
policy = np.argmax(Q, axis=1)


# Map actions to their corresponding names
actions = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}


# Print the policy
print("\nFinal Policy (Best Action for Each State):")
for state in range(n_states):
    print(f"State {state}: {actions[policy[state]]}")


# Visual representation of the Q-table from SARSA
fig, ax = plt.subplots(figsize=(8, 8))


# Plot the Q-values for each action
cax = ax.matshow(Q, cmap="coolwarm")


# Set grid labels
ax.set_xticks(np.arange(0, n_actions, 1))
ax.set_yticks(np.arange(0, n_states, 1))


# Add Q-values in the plot
for i in range(n_states):
    for j in range(n_actions):
        ax.text(j, i, f'{Q[i, j]:.2f}', va='center', ha='center')


# Set axis labels
ax.set_xlabel('Actions')
ax.set_ylabel('States')


# Add color bar to represent Q-values
fig.colorbar(cax)


# Add a title
plt.title('SARSA Q-Values Visualization for Frozen Lake')


# Show the plot
plt.show()
Temporal difference evaluation.py:
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# Custom rewards for each state
custom_rewards = np.array([
    0, -0.5, 0, -1,    # Row 1
    0, -0.2, 0, -1,    # Row 2
    0, -1, 0, -1,      # Row 3
    0, 0, 0, 1         # Row 4 (Goal state has a positive reward)
])


# Create the FrozenLake environment with deterministic behavior
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")


# Parameters
alpha = 0.1
gamma = 0.9
episodes = 10  # Number of episodes for demonstration purposes


# Initialize value function
V = np.zeros(env.observation_space.n)


# Optimal Policy for 4x4 FrozenLake
def optimal_policy(state):
    """Optimal policy for 4x4 FrozenLake with corrected actions for state 13 and 14."""
    policy_map = {
        0: 1,  # Move right
        1: 1,  # Move right
        2: 1,  # Move right
        3: 2,  # Move down
        4: 1,  # Move right
        5: 2,  # Move down
        6: 1,  # Move right
        7: 2,  # Move down
        8: 2,  # Move down
        9: 1,  # Move right
        10: 1, # Move right
        11: 2, # Move down
        12: 1, # Move right
        13: 2, # Move down
        14: 2, # Move down to reach goal state 15
        15: 0  # Goal state, no move needed
    }
    return policy_map[state]






# TD(0) Algorithm with continuous rendering and custom rewards
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    print(f"\n--- Episode {episode + 1} ---")


    while not done:
        # Get the frame from the environment
        frame = env.render()


        # Display the frame using Matplotlib
        plt.imshow(frame)
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.5)  # Adjust the pause time as needed
        plt.close()


        action = optimal_policy(state)
        print(f"State: {state}, Action: {action}")  # Debugging: print current state and action
        next_state, reward, done, _, _ = env.step(action)
        print(f"Next State: {next_state}, Reward: {reward}, Done: {done}")  # Debugging: print the next state and whether the episode is done


        # Apply custom rewards
        reward = custom_rewards[next_state]


        # Calculate TD(0) update
        old_value = V[state]
        td_target = reward + gamma * V[next_state]
        td_error = td_target - old_value
        V[state] = old_value + alpha * td_error
       
        # Print the TD(0) update step-by-step
        print(f"Old Value: {old_value:.4f}, TD Target: {td_target:.4f}, TD Error: {td_error:.4f}")
        print(f"Updated Value for State {state}: {V[state]:.4f}")
        print("Current Value Function:")
        print(V.reshape((4, 4)))
        print("\n" + "-"*50 + "\n")
       
        # Pause after each sample to allow for explanation
        input("Press Enter to continue to the next sample...")


        state = next_state


    # After each episode, show the value function
    print("Value function after this episode:")
    print(V.reshape((4, 4)))


# Final visualization using Matplotlib
V_grid = V.reshape((4, 4))
plt.figure(figsize=(6, 6))
plt.imshow(V_grid, cmap='coolwarm', interpolation='none')
plt.colorbar(label="State Value")
plt.title("State Value Function for FrozenLake with Custom Rewards")
plt.show()


env.close()


