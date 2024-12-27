import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from network import SimpleCNN

class PPO:
    def __init__(self, env):        
        self._init_hyperparameters()

        self.env = env
        self.obs_dim = [env.observation_space.shape[0], env.observation_space.shape[1]]
        self.act_dim = 6

        self.actor = SimpleCNN(self.obs_dim, self.act_dim)
        self.critic = SimpleCNN(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size = (self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_timesteps):
        ''' Function that follow every step of PPO-clip algorithm '''
    	
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        
        t_so_far = 0                            # timesteps simulated so far
        i_so_far = 0                            # iterations ran so far
        while t_so_far < total_timesteps:       # the cycle end after a certain number of epochs
            # to obtain all the information we have to rollout the data from the enviroment
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)
            i_so_far *= 1

            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph = True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

    def evaluate(self, batch_obs, batch_acts):
        ''' Compute the current Value function and the log probability of the Action '''

        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(batch_acts)
        return V, log_prob
    
    def rollout(self):
        ''' Function that unpack the data from the enviroment '''

        batch_obs = []          # batch observations
        batch_acts = []         # batch actions
        batch_log_probs = []    # log probability of each action
        batch_rews = []         # batch rewards
        batch_rtgs = []         # batch reward-to-go
        batch_lens = []         # episodic lenghts in batch

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs, info = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, trunc, info = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done or trunc:
                    break
        
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(batch_obs, dtype = torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()
    
    def compute_rtgs(self, batch_rews):

        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def _init_hyperparameters(self):
        ''' Here are defined the parameters that are used in the learning process '''

        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005





