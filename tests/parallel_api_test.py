from __future__ import annotations
from generals.env import pz_generals
from generals.agents import RandomAgent
from generals.map import Mapper
import warnings

import numpy as np

from pettingzoo.test.api_test import missing_attr_warning
from pettingzoo.utils.conversions import (
    aec_to_parallel_wrapper,
    parallel_to_aec_wrapper,
    turn_based_aec_to_parallel_wrapper,
)
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
from pettingzoo.utils.wrappers import BaseWrapper


def sample_action(
    env: ParallelEnv[AgentID, ObsType, ActionType],
    obs: dict[AgentID, ObsType],
    agent: AgentID,
) -> ActionType:
    agent_obs = obs[agent]
    # custom action action sampler
    if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
        mask = agent_obs["action_mask"]
        # pick random index of the mask with a 1, it should be 3 numbers
        valid_actions = np.argwhere(mask == 1)
        if len(valid_actions) == 0:
            return np.array([1, 0, 0, 0, 0])
        action_index = np.random.choice(len(valid_actions))
        action = np.array([1])
        action = np.append(action, valid_actions[action_index])
        # append 1 or 0 randomly to the action (to say whether to send half of troops or all troops)
        action = np.append(action, np.random.choice([0, 1]))
        return action
    return env.action_space(agent).sample()


def parallel_api_test(par_env: ParallelEnv, num_cycles=1000):
    par_env.max_cycles = num_cycles

    if not hasattr(par_env, "possible_agents"):
        warnings.warn(missing_attr_warning.format(name="possible_agents"))

    assert not isinstance(par_env.unwrapped, aec_to_parallel_wrapper)
    assert not isinstance(par_env.unwrapped, parallel_to_aec_wrapper)
    assert not isinstance(par_env.unwrapped, turn_based_aec_to_parallel_wrapper)
    assert not isinstance(par_env.unwrapped, BaseWrapper)

    # checks that reset takes arguments seed and options
    par_env.reset(seed=0, options={"options": 1})

    t = 0
    while t < 100_000:
        obs, infos = par_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(infos, dict)
        # Note: obs and info dicts must contain all AgentIDs, but can also have other additional keys (e.g., "common")
        assert set(par_env.agents).issubset(set(obs.keys()))
        assert set(par_env.agents).issubset(set(infos.keys()))
        terminated = {agent: False for agent in par_env.agents}
        truncated = {agent: False for agent in par_env.agents}
        live_agents = set(par_env.agents[:])
        has_finished = set()
        for _ in range(num_cycles):
            t += 1
            actions = {
                agent: sample_action(par_env, obs, agent)
                for agent in par_env.agents
                if (
                    (agent in terminated and not terminated[agent])
                    or (agent in truncated and not truncated[agent])
                )
            }
            obs, rew, terminated, truncated, info = par_env.step(actions)
            for agent in par_env.agents:
                assert agent not in has_finished, "agent cannot be revived once dead"

                if agent not in live_agents:
                    live_agents.add(agent)

            assert isinstance(obs, dict)
            assert isinstance(rew, dict)
            assert isinstance(terminated, dict)
            assert isinstance(truncated, dict)
            assert isinstance(info, dict)

            keys = "observation reward terminated truncated info".split()
            vals = [obs, rew, terminated, truncated, info]
            for k, v in zip(keys, vals):
                key_set = set(v.keys())
                if key_set == live_agents:
                    continue
                if len(key_set) < len(live_agents):
                    warnings.warn(f"Live agent was not given {k}")
                else:
                    warnings.warn(f"Agent was given {k} but was dead last turn")

            if hasattr(par_env, "possible_agents"):
                assert set(par_env.agents).issubset(
                    set(par_env.possible_agents)
                ), "possible_agents defined but does not contain all agents"

                has_finished |= {
                    agent
                    for agent in live_agents
                    if terminated[agent] or truncated[agent]
                }
                if not par_env.agents and has_finished != set(par_env.possible_agents):
                    warnings.warn(
                        "No agents present but not all possible_agents are terminated or truncated"
                    )
            elif not par_env.agents:
                warnings.warn("No agents present")

            for agent in par_env.agents:
                assert par_env.observation_space(agent) is par_env.observation_space(
                    agent
                ), "observation_space should return the exact same space object (not a copy) for an agent.\
                    Consider decorating your observation_space(self, agent) method with @functools.lru_cache(maxsize=None)"
                assert par_env.action_space(agent) is par_env.action_space(
                    agent
                ), "action_space should return the exact same space object (not a copy) for an agent\
                (ensures that action space seeding works as expected). Consider decorating your action_space(self, agent)\
                method with @functools.lru_cache(maxsize=None)"

            agents_to_remove = {
                agent for agent in live_agents if terminated[agent] or truncated[agent]
            }
            live_agents -= agents_to_remove

            assert (
                set(par_env.agents) == live_agents
            ), f"{par_env.agents} != {live_agents}"

            if len(live_agents) == 0:
                break
    print(f"Total cycles: {t}")
    print("Passed Parallel API test")

if __name__ == "__main__":
    mapper = Mapper()
    agent1 = RandomAgent(name="A")
    agent2 = RandomAgent(name="B")
    agents = {
        agent1.name: agent1,
        agent2.name: agent2,
    }
    env = pz_generals(mapper, agents)
    # test the environment with parallel_api_test
    import time
    start = time.time()
    n_cycles = 1000
    parallel_api_test(env, num_cycles=n_cycles)
    print(f"Time for {n_cycles} cycles: {time.time() - start}")
    env.close()
