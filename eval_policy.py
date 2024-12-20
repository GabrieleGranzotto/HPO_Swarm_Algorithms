import cv2
import gymnasium as gym
from ppo import PPO

if __name__ == "__main__":
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
    env.reset()
    print(env.action_space.n)
    #model = PPO(env)
    #model.learn(10000)
    env.close
