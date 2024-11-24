import numpy as np

def update(w, x, x_t1, reward, alpha, gamma, done):
    """
    Return the updated weights vector w_t1
    @param w: the weights vector before the update
    @param x: the feature vector observed at t
    @param x_t1: the feature vector observed at t+1
    @param reward: the reward observed after the action
    @param alpha: the step size (learning rate)
    @param gamma: the discount factor
    @param done: boolean True if the state is terminal
    @return: w_t1 the weights vector at t+1
    """
    if done:
        # Terminal state, no future state
        w_t1 = w + alpha * ((reward - np.dot(x, w)) * x)
    else:
        # Non-terminal state, use next state's feature vector
        w_t1 = w + alpha * ((reward + (gamma * np.dot(x_t1, w)) - np.dot(x, w)) * x)
    
    return w_t1

# Example of using this function
# Initialize weights vector
w = np.random.randn(3)  # Assuming we have 3 features
# Feature vectors for state t and t+1
x = np.array([0.5, 1.0, -0.5])
x_t1 = np.array([0.6, 1.1, -0.4])
reward = 1.0
alpha = 0.01
gamma = 0.99
done = False  # Non-terminal state

# Update weights
updated_w = update(w, x, x_t1, reward, alpha, gamma, done)
print("Updated weights:", updated_w)
