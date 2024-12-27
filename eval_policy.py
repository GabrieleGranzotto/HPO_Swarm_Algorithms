import cv2
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from ppo2 import PPO

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)
    model = PPO(env)
    model.learn(10000)
    env.close
