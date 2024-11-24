import gymnasium as gym
import time
env = gym.make(
    "FrozenLake-v1",
    render_mode="human",
    map_name="4x4",            # Optional argument to specify map size
    is_slippery=True,          # Optional argument to make the environment slippery
)


# Policy examples
def always_right_policy(state):
    return 2  # Right


def random_policy(state):
    return env.action_space.sample()


def greedy_policy(state):
    if state in [0, 1, 2, 4, 5, 6, 8, 9, 10]:  # Move right if possible
        return 2  # Right
    else:
        return 1  # Down


def avoid_holes_policy(state):
    hole_states = [5, 7, 11, 12]
    if state in hole_states:
        return 3  # Up
    if state % 4 == 0:  # If in the leftmost column, move right
        return 2  # Right
    else:
        return 1  # Down


def zigzag_policy(state):
    if state % 2 == 0:
        return 2  # Right on even states
    else:
        return 1  # Down on odd states


# Function to run the simulation
def run_simulation(env, policy, num_steps=30, delay=1):
    state, _ = env.reset()
    env.render()
    for step in range(num_steps):
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(delay)
       
        # Print detailed information at each step
        print(f"Step: {step + 1}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Next State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
        print("---------")
       
        state = next_state
        if terminated or truncated:
            print(f"Episode finished after {step + 1} timesteps")
            print(f"Final Reward: {reward}")
            break
    #env.close()


# Run the simulation with different policies




print("Testing Random Policy")
run_simulation(env, random_policy)


print("Testing Greedy Policy")
run_simulation(env, greedy_policy)


print("Testing Avoid Holes Policy")
run_simulation(env, avoid_holes_policy)


print("Testing Zigzag Policy")
run_simulation(env, zigzag_policy)


print("Testing Always Right Policy")
run_simulation(env, always_right_policy)
