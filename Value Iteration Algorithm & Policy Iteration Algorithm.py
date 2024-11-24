import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
# SFFF
# FHFH
# FFFH
# HFFG


policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}


#env = gym.make('FrozenLake-v1', is_slippery=True, render_mode="human")  # Updated to FrozenLake-v1 and added render_mode
env = gym.make('FrozenLake-v1', is_slippery=True)
n_games = 100000
win_pct = []
scores = []


for i in range(n_games):
    done = False
    obs, info = env.reset()  # Updated reset() to return obs and info
    score = 0
    while not done:
        #env.render()  # Render the environment at each step
        action = policy[obs]  # Select action based on the predefined policy
        obs, reward, done, truncated, info = env.step(action)  # Updated step() to return truncated
        score += reward
    scores.append(score)
    if i % 1000 == 0:
        average = np.mean(scores[-1000:])
        win_pct.append(average)


plt.plot(win_pct)
plt.show()


env.close()  # Close the environment when done





import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
# SFFF
# FHFH
# FFFH
# HFFG


policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}


env = gym.make('FrozenLake-v1', is_slippery=True)


# Function to play the game using a given policy and return the win percentages over time
def play_games(env, policy, n_games=100000):
    win_pct = []
    scores = []
    for i in range(n_games):
        done = False
        obs, info = env.reset()
        score = 0
        while not done:
            action = policy[obs]
            obs, reward, done, truncated, info = env.step(action)
            score += reward
        scores.append(score)
        if i % 1000 == 0:
            average = np.mean(scores[-1000:])
            win_pct.append(average)
    return win_pct


# Value Iteration Algorithm
def value_iteration(env, gamma=1.0, max_iterations=1000, theta=1e-20):
    value_table = np.zeros(env.observation_space.n)
    P = env.unwrapped.P  # Accessing the transition probabilities using env.unwrapped.P
    for i in range(max_iterations):
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_state_reward = []
                for prob, next_state, reward, done in P[state][action]:
                    next_state_reward.append(prob * (reward + gamma * updated_value_table[next_state]))
                Q_value.append(np.sum(next_state_reward))
            value_table[state] = max(Q_value)
        if np.sum(np.fabs(updated_value_table - value_table)) <= theta:
            break
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        Q_value = []
        for action in range(env.action_space.n):
            next_state_reward = []
            for prob, next_state, reward, done in P[state][action]:
                next_state_reward.append(prob * (reward + gamma * value_table[next_state]))
            Q_value.append(np.sum(next_state_reward))
        policy[state] = np.argmax(Q_value)
    return policy


# Policy Iteration Algorithm
def policy_iteration(env, gamma=1.0, max_iterations=1000):
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))
    value_table = np.zeros(env.observation_space.n)
    P = env.unwrapped.P  # Accessing the transition probabilities using env.unwrapped.P
    for i in range(max_iterations):
        while True:
            updated_value_table = np.copy(value_table)
            for state in range(env.observation_space.n):
                action = policy[state]
                value_table[state] = sum([prob * (reward + gamma * updated_value_table[next_state])
                                          for prob, next_state, reward, done in P[state][action]])
            if np.sum(np.fabs(updated_value_table - value_table)) <= 1e-20:
                break
       
        stable_policy = True
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                Q_value.append(sum([prob * (reward + gamma * value_table[next_state])
                                    for prob, next_state, reward, done in P[state][action]]))
            best_action = np.argmax(Q_value)
            if policy[state] != best_action:
                stable_policy = False
            policy[state] = best_action
       
        if stable_policy:
            break
    return policy


# Play the game with the predefined policy
win_pct_predefined = play_games(env, policy)


# Solve the problem using Value Iteration
value_iter_policy = value_iteration(env)
win_pct_value_iteration = play_games(env, value_iter_policy)


# Solve the problem using Policy Iteration
policy_iter_policy = policy_iteration(env)
win_pct_policy_iteration = play_games(env, policy_iter_policy)


# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(range(0, len(win_pct_predefined) * 1000, 1000), win_pct_predefined, label='Predefined Policy', color='blue')
plt.plot(range(0, len(win_pct_value_iteration) * 1000, 1000), win_pct_value_iteration, label='Value Iteration', color='green')
plt.plot(range(0, len(win_pct_policy_iteration) * 1000, 1000), win_pct_policy_iteration, label='Policy Iteration', color='red')
plt.xlabel('Games Played')
plt.ylabel('Win Percentage')
plt.title('Win Percentage Over Time for Different Policies')
plt.legend()
plt.grid(True)
plt.show()


env.close()
