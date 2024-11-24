
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Define the neural network policy
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


# Define the reinforce algorithm
class REINFORCE:
    def __init__(self, env, policy_network, lr=0.01, gamma=0.99):
        self.env = env
        self.policy_network = policy_network
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.episode_rewards = []  # Track rewards per episode


    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy_network(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)


        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)


        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward.unsqueeze(0))  # Ensure reward is one-dimensional
        policy_loss = torch.stack(policy_loss).sum()


        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


    def train(self, num_episodes, plot_interval=10):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            rewards = []
            log_probs = []


            while True:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)


                rewards.append(reward)
                log_probs.append(log_prob)


                if done:
                    self.update_policy(rewards, log_probs)
                    total_reward = sum(rewards)
                    self.episode_rewards.append(total_reward)
                    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
                    break


                state = next_state


            # Plot the performance every few episodes
            if (episode + 1) % plot_interval == 0:
                self.plot_performance(plot_interval)


    def plot_performance(self, plot_interval):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label='Episode Reward')
        # Running average of rewards
        if len(self.episode_rewards) >= plot_interval:
            running_avg = np.convolve(self.episode_rewards, np.ones(plot_interval) / plot_interval, mode='valid')
            plt.plot(range(plot_interval - 1, len(self.episode_rewards)), running_avg, label='Running Average')
       
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Performance Over Episodes')
        plt.legend()
        plt.show()


# Main code for environment setup and training
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n


    policy_network = PolicyNetwork(state_dim, action_dim)
    agent = REINFORCE(env, policy_network)


    agent.train(num_episodes=1000)
    env.close()
