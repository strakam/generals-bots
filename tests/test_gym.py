import gymnasium as gym
import gymnasium.utils.env_checker as env_checker

from generals.agents import AgentFactory


def test_gym_runs():
    npc = AgentFactory.make_agent("random")

    env = gym.make(
        "gym-generals-v0",
        npc=npc,
    )
    env_checker.check_env(env.unwrapped)
    print("Gymnasium check passed!")
