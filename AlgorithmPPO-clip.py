# Algorithm PPO-clip
# 
# Input: initial policy parameters theta_0, initial value function parameter phi_0
# for k=0,1,2,... do
#   collect set of trajectories D_k = {tao_i} by running policy pi_k = pi(theta_k) in the enviroment
#   compute reward-to-go R_t hat
#   compute advantage estimates A_t hat (using any method of advantage estimation) based on current value function V_psi_k
#   update the policy by maximizing the PPO-clip objective typically via stochastic gradient ascent with Adam
#   fit value function by regression on mean-squared error typically via some gradient descent algorithm
# end for

import torch
from network import Network

class PPO:
    def __init__(self, env):
        # call the hyperparameters
        self._hyperparameters()
        # call the environment
        self.env = env
        # take the observation dimension
        self.obs_dim = [env.observation_space.shape[0], env.observation_space.shape[1]]
        self.act_dim = env.action_space.n
        # call the neural network of actor and critic
        self.actor = Network(self.obs_dim, self.act_dim)
        self.critic = Network(self.obs_dim, 1)
        # I need to understand this code! I think it is useless in my case
        self.cov_var = torch.full(size = (self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self):
        pass
    def evaluate(self):
        pass
    def _hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005