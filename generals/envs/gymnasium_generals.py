from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from generals.agents import Agent, RandomAgent
from generals.core.environment import Action, Environment
from generals.core.grid import GridFactory
from generals.core.observation import Observation
from generals.rewards.reward_fn import RewardFn


class GymnasiumGenerals(gym.Env):
    def __init__(
        self,
        grid_factory: GridFactory = None,
        npc: Agent = None,
        agent: Agent = None,
        truncation: int = None,
        reward_fn: RewardFn = None,
        to_render: bool = False,
    ):
        if npc is None:
            npc = RandomAgent()
        assert isinstance(npc, Agent), "NPC must be an instance of Agent class."
        assert agent.id != npc.id, "Agent ids must be unique - you can pass custom ids to agent constructors."

        self.npc = npc
        self.agent_id = "Agent" if agent is None else agent.id
        self.agent_ids = [self.agent_id, self.npc.id]

        self.environment = Environment(
            agent_ids=[self.agent_id, npc.id],
            grid_factory=grid_factory,
            truncation=truncation,
            reward_fn=reward_fn,
            to_render=to_render,
        )
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()

    def set_observation_space(self) -> spaces.Space:
        dims = self.environment.grid_dims
        grid_multi_binary = spaces.MultiBinary(dims)
        grid_discrete = np.ones(dims, dtype=int) * Environment.max_army_size

        return spaces.Dict(
            {
                "armies": spaces.MultiDiscrete(grid_discrete),
                "generals": grid_multi_binary,
                "cities": grid_multi_binary,
                "mountains": grid_multi_binary,
                "neutral_cells": grid_multi_binary,
                "owned_cells": grid_multi_binary,
                "opponent_cells": grid_multi_binary,
                "fog_cells": grid_multi_binary,
                "structures_in_fog": grid_multi_binary,
                "owned_land_count": spaces.Discrete(Environment.max_land_owned),
                "owned_army_count": spaces.Discrete(Environment.max_army_size),
                "opponent_land_count": spaces.Discrete(Environment.max_land_owned),
                "opponent_army_count": spaces.Discrete(Environment.max_army_size),
                "timestep": spaces.Discrete(Environment.max_timestep),
                "priority": spaces.Discrete(2),
            }
        )

    def set_action_space(self) -> spaces.Space:
        dims = self.environment.grid_dims
        return spaces.MultiDiscrete([2, dims[0], dims[1], 4, 2])

    def render(self):
        self.environment.render()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        # Reset the parent Gymnasium.env first.
        super().reset(seed=seed)

        # Provide the np.random.Generator instance created in Env.reset()
        # as opposed to creating a new one with the same seed.
        observation, info = self.environment.reset_from_gymnasium(rng=self.np_random, options=options)

        # In case the grid-dims have changed.
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()

        return observation, info

    def step(self, action: Action) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        # Get action of NPC
        npc_observation = self.environment.agent_observation(self.npc.id)
        npc_action = self.npc.act(npc_observation)
        actions = {self.agent_id: action, self.npc.id: npc_action}

        # Step the underlying environment.
        observations, rewards, terminated, truncated, infos = self.environment.step(actions)

        # Only provide information relevant to the main-agent, not the npc.
        return (
            observations[self.agent_id],
            rewards[self.agent_id],
            terminated,
            truncated,
            infos[self.agent_id],
        )

    def close(self) -> None:
        self.environment.close()
