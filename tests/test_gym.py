import gymnasium as gym
import gymnasium.utils.env_checker as env_checker

from generals.agents import RandomAgent


def test_gym_runs():
    env = gym.make(
        "gym-generals-v0",
        npc=RandomAgent(id="thing-1"),
        agent=RandomAgent(id="thing-2"),
    )
    env_checker.check_env(env.unwrapped)
    print("Gymnasium check passed!")
