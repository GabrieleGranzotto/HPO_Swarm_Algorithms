import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5
BATCH_SIZE = 64
EPOCHS = 4
STEPS_PER_UPDATE = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CNNActorCritic, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        n_features = self._get_feature_size(input_shape)
        
        self.actor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_feature_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            return self.feature_extractor(x).view(1, -1).size(1)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

def compute_gae(rewards, values, dones):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + GAMMA * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages

def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(EPOCHS):
        for start in range(0, len(states), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_states = states[start:end]
            batch_actions = actions[start:end]
            batch_old_log_probs = old_log_probs[start:end]
            batch_returns = returns[start:end]
            batch_advantages = advantages[start:end]

            logits, values = model(batch_states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)

            entropy = dist.entropy().mean()
            
            ratio = (new_log_probs - batch_old_log_probs).exp()
            surrogate1 = ratio * batch_advantages
            surrogate2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            value_loss = (batch_returns - values.squeeze(-1)).pow(2).mean()

            loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def main():
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = CNNActorCritic(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    state = env.reset()
    state = torch.tensor(state[0], dtype=torch.float32).to(device).permute(2, 0, 1).unsqueeze(0)

    episode_rewards = []
    for _ in range(1000):
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        
        for _ in range(STEPS_PER_UPDATE):
            logits, value = model(state)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_state, reward, done, _, _ = env.step(action.item())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            log_probs.append(dist.log_prob(action))

            state = torch.tensor(next_state, dtype=torch.float32).to(device).permute(2, 0, 1).unsqueeze(0)

            if done:
                episode_rewards.append(sum(rewards))
                state = env.reset()
                state = torch.tensor(state[0], dtype=torch.float32).to(device).permute(2, 0, 1).unsqueeze(0)

        _, next_value = model(state)
        values.append(next_value)
        
        advantages = compute_gae(rewards, values, dones)
        returns = advantages + values[:-1]
        
        states = torch.cat(states)
        actions = torch.cat(actions)
        old_log_probs = torch.cat(log_probs)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages)

if __name__ == "__main__":
    main()
