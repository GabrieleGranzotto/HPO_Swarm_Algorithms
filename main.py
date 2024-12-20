import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

# Define the CNN-based Actor-Critic model
class ActorCriticCNN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(ActorCriticCNN, self).__init__()
        # Shared CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )
        # Actor and Critic heads
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# Hyperparameters
learning_rate = 2.5e-4
gamma = 0.99
epsilon_clip = 0.2
k_epochs = 4
entropy_coeff = 0.01

# PPO Class
class PPO:
    def __init__(self, input_channels, num_actions):
        self.policy = ActorCriticCNN(input_channels, num_actions).cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()

    def compute_loss(self, states, actions, log_probs, returns, advantages):
        # Forward pass through the model
        logits, state_values = self.policy(states)
        dist = Categorical(logits=logits)
        
        # Compute new log probabilities
        new_log_probs = dist.log_prob(actions)
        
        # Compute policy ratio
        ratio = torch.exp(new_log_probs - log_probs)
        
        # Compute clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Compute critic loss
        critic_loss = self.mse_loss(state_values.squeeze(), returns)
        
        # Compute entropy bonus
        entropy = dist.entropy().mean()
        
        # Combine losses
        total_loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy
        return total_loss

    def update(self, memory):
        states, actions, log_probs, returns, advantages = memory.sample()
        states = torch.tensor(states, dtype=torch.float32).cuda()
        actions = torch.tensor(actions, dtype=torch.int64).cuda()
        log_probs = torch.tensor(log_probs, dtype=torch.float32).cuda()
        returns = torch.tensor(returns, dtype=torch.float32).cuda()
        advantages = torch.tensor(advantages, dtype=torch.float32).cuda()

        for _ in range(k_epochs):
            loss = self.compute_loss(states, actions, log_probs, returns, advantages)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Memory Class for PPO
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def sample(self):
        # Compute discounted rewards and advantages
        rewards, advantages = self.compute_gae()
        return self.states, self.actions, self.log_probs, rewards, advantages

    def compute_gae(self, gamma=0.99, lambda_=0.95):
        rewards = []
        advantages = []
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * (1 - self.dones[t]) * running_add - self.values[t]
            running_add = delta + gamma * lambda_ * running_add
            rewards.insert(0, running_add + self.values[t])
            advantages.insert(0, running_add)
        return rewards, advantages

    def clear(self):
        self.states, self.actions, self.log_probs, self.rewards, self.values, self.dones = [], [], [], [], [], []

# Main Training Loop
env = gym.make("ALE/SpaceInvaders-v5")
ppo = PPO(input_channels=3, num_actions=env.action_space.n)
memory = Memory()

for episode in range(1000):
    state, _ = env.reset()
    state = torch.tensor(state.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).cuda()  # Convert to tensor
    done = False
    episode_reward = 0

    while not done:
        logits, value = ppo.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_state, reward, done, _, _ = env.step(action.item())
        next_state = torch.tensor(next_state.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).cuda()
        
        memory.add(state.cpu().numpy(), action.item(), log_prob.item(), reward, value.item(), done)
        state = next_state
        episode_reward += reward

    # Update PPO after an episode
    ppo.update(memory)
    memory.clear()

    print(f"Episode {episode+1}: Reward = {episode_reward}")
