import gymnasium as gym
from ppo import PPO
env = gym.make("BipedalWalker-v3", render_mode="human")
model = PPO(env)
model.learn(10000)
env.close()
    