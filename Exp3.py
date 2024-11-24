import numpy as np
import gym
import time


# Initialize the Frozen Lake environment
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')


# Parameters
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.99
theta = 1e-6


# Initialize the policy and value function
policy = np.zeros(n_states, dtype=int)
value_function = np.zeros(n_states)


def policy_evaluation(policy):
    while True:
        delta = 0
        for s in range(n_states):
            v = value_function[s]
            action = policy[s]
            # Bellman equation
            value_function[s] = sum([prob * (reward + gamma * value_function[next_state])
                                    for prob, next_state, reward, _ in env.P[s][action]])
            delta = max(delta, abs(v - value_function[s]))
        if delta < theta:
            break


def policy_improvement():
    policy_stable = True
    for s in range(n_states):
        old_action = policy[s]
        # Improve policy
        policy[s] = np.argmax([sum([prob * (reward + gamma * value_function[next_state])
                                    for prob, next_state, reward, _ in env.P[s][a]])
                              for a in range(n_actions)])
        if old_action != policy[s]:
            policy_stable = False
    return policy_stable


def policy_iteration():
    global policy
    iteration = 0
    while True:
        print(f"Policy Iteration: {iteration}")
       
        # Policy Evaluation
        policy_evaluation(policy)
       
        # Policy Improvement
        stable = policy_improvement()
       
        if stable:
            break
        iteration += 1


    return policy


# Compute the optimal policy
optimal_policy = policy_iteration()
print("Optimal Policy:", optimal_policy)


# Run the optimal policy to visualize its performance
obs = env.reset()
done = False


print("Starting visualization:")
while not done:
    # Ensure the observation is in a format that can be used as an index
    if isinstance(obs, np.ndarray):
        obs = int(obs[0])
    elif isinstance(obs, tuple):
        obs = int(obs[0])
    elif isinstance(obs, int):
        pass  # No conversion needed
    else:
        raise ValueError(f"Unexpected observation format: {type(obs)}")


    # Retrieve the action from the optimal policy
    action = optimal_policy[obs]


    # Step in the environment and render
    result = env.step(action)
    print("Step result:", result)  # Print the result to understand its structure


    # Unpack the five values
    obs, reward, done, terminated, info = result


    # Handle the 'terminated' value if needed (optional)
    if terminated:
        done = True


    env.render()  # Render the environment
    time.sleep(1)  # Slow down rendering for visualization


env.close()


