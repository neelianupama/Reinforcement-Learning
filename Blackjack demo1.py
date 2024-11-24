#Blackjack demo1:
import gymnasium as gym


# Initialize the Blackjack environment
env = gym.make('Blackjack-v1', sab=True)


# Number of episodes to demonstrate
num_demos = 5


# Print the initial state at the start of a few episodes
for episode in range(num_demos):
    # Reset the environment to start a new game
    initial_state, _ = env.reset()


    # Print the initial state
    print(f"Episode {episode + 1}: Initial State: {initial_state}")
Blackjack demo 2:
import random


# Number of random state-action pairs to generate
num_samples = 5


# Generate and print random state-action pairs
for i in range(num_samples):
    # Randomly generate a state
    state = (
        random.randint(4, 21),  # Player's current sum (between 4 and 21)
        random.randint(1, 10),  # Dealer's showing card (between 1 and 10)
        random.choice([True, False])  # Whether the player has a usable ace
    )
   
    # Randomly generate an action
    action = random.choice([0, 1])  # 0 = Stick, 1 = Hit
   
    # Print the generated state-action pair
    print(f"Sample {i + 1}: State: {state}, Action: {'Stick' if action == 0 else 'Hit'}")
  
#Black jack mc1:
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt


# Initialize the Blackjack environment
env = gym.make('Blackjack-v1', sab=True)


# Set parameters for the algorithm
num_episodes = 500000
epsilon = 0.3  # Probability of selecting a random action


# Initialize Q(s,a) and returns(s,a)
Q = {}
returns = {}


# Initialize Q-values and returns as empty dictionaries
def initialize_Q():
    for player_sum in range(4, 22):  # Player's current sum
        for dealer_showing in range(1, 11):  # Dealer's showing card
            for usable_ace in [True, False]:  # Whether the player has a usable ace
                for action in [0, 1]:  # Possible actions: stick (0), hit (1)
                    Q[((player_sum, dealer_showing, usable_ace), action)] = 0.0
                    returns[((player_sum, dealer_showing, usable_ace), action)] = []


initialize_Q()


# Function to choose action using ε-greedy policy derived from Q
def epsilon_greedy_policy(state, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:  # Exploration: choose a random action
        return random.choice([0, 1])
    else:  # Exploitation: choose the best action based on current Q values
        return np.argmax([Q[(state, a)] for a in [0, 1]])


# Main loop for Monte Carlo Control with ε-Greedy
rewards_per_episode = []


for episode in range(num_episodes):
    # Reset the environment to start a new episode
    state, _ = env.reset()
   
    state_action_pairs = []
    done = False
    total_reward = 0
   
    # Generate an episode following the ε-greedy policy
    while not done:
        action = epsilon_greedy_policy(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state_action_pairs.append((state, action))
        state = next_state
   
    # For each pair (S, A) appearing in the episode:
    for state, action in state_action_pairs:
        sa_pair = (state, action)
       
        # Append the return G to the list of returns for this pair (S, A)
        returns[sa_pair].append(total_reward)
       
        # Set Q(S, A) to the average of all returns that followed from state-action pair (S, A)
        Q[sa_pair] = np.mean(returns[sa_pair])
   
    rewards_per_episode.append(total_reward)


    # Print progress every 100,000 episodes
    if (episode + 1) % 100000 == 0:
        average_reward = np.mean(rewards_per_episode[-100000:])
        print(f"Episode {episode + 1}/{num_episodes} - Average Reward: {average_reward}")


# Plotting the performance
plt.plot(np.convolve(rewards_per_episode, np.ones(1000)/1000, mode='valid'))
plt.xlabel('Episode')
plt.ylabel('Average Reward (1000 Episode Window)')
plt.title('Performance of Blackjack Agent using MC Control with ε-Greedy')
plt.show()


# Display learned Q-values for a sample state
sample_state = (18, 4, True)
print(f"Q-values for state {sample_state}:")
print(f"Stick: {Q[(sample_state, 0)]}, Hit: {Q[(sample_state, 1)]}")


# Optionally, test the learned policy
state, _ = env.reset()
done = False
while not done:
    action = epsilon_greedy_policy(state, epsilon=0.0)  # Use greedy policy (no exploration)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print(f"State: {state}, Action: {'Stick' if action == 0 else 'Hit'}, Reward: {reward}")
    state = next_state
  
#Blackjack mc2.py:
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt


# Initialize the Blackjack environment
env = gym.make('Blackjack-v1', sab=True)


# Initialize Q and returns dictionaries
Q = {}
returns = {}


def initialize_Q():
    """Initialize Q-values and returns dictionaries."""
    for player_sum in range(4, 22):  # Player's hand total
        for dealer_showing in range(1, 11):  # Dealer's visible card
            for usable_ace in [True, False]:  # Usable Ace status
                for action in [0, 1]:  # 0 = Stick, 1 = Hit
                    Q[((player_sum, dealer_showing, usable_ace), action)] = 0.0
                    returns[((player_sum, dealer_showing, usable_ace), action)] = []


initialize_Q()


# Main loop for episodes
num_episodes = 500000  # Total number of episodes to run
rewards_per_episode = []


for episode in range(num_episodes):
    # Reset the environment at the beginning of the episode
    state, _ = env.reset()
   
    # Exploring start: Override the initial state with a random state and action
    state = (
        random.randint(4, 21),  # Player's current sum
        random.randint(1, 10),  # Dealer's showing card
        random.choice([True, False])  # Usable Ace status
    )
    action = random.choice([0, 1])  # Random initial action: 0 = Stick, 1 = Hit
    state_action_pairs = [(state, action)]
   
    done = False
    total_reward = 0
   
    while not done:
        # Take action and observe the outcome
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
       
        if not done:
            # Choose the next action greedily based on Q-values
            action = np.argmax([Q[(next_state, a)] for a in [0, 1]])
            state_action_pairs.append((next_state, action))


    # After the episode ends, update Q-values for the visited state-action pairs
    for state, action in state_action_pairs:
        # Append the observed return (total reward) to the returns list
        returns[(state, action)].append(total_reward)
       
        # Update Q-value to be the average of returns
        Q[(state, action)] = np.mean(returns[(state, action)])


    # Store the total reward for this episode
    rewards_per_episode.append(total_reward)


    # Print progress every 100,000 episodes
    if (episode + 1) % 100000 == 0:
        average_reward = np.mean(rewards_per_episode[-100000:])
        print(f"Episode {episode + 1}/{num_episodes} - Average Reward: {average_reward}")


# Plotting the performance
plt.plot(np.convolve(rewards_per_episode, np.ones(1000)/1000, mode='valid'))
plt.xlabel('Episode')
plt.ylabel('Average Reward (1000 Episode Window)')
plt.title('Performance of Blackjack Agent using MC Control with Exploring Starts')
plt.show()


# Optionally, display learned Q-values for a specific state
sample_state = (18, 5, True)
print(f"Q-values for state {sample_state}:")
print(f"Stick: {Q[(sample_state, 0)]}, Hit: {Q[(sample_state, 1)]}")
