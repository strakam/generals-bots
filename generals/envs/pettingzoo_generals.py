import functools
from collections.abc import Callable
from copy import deepcopy
from typing import Any, TypeAlias

import pettingzoo  # type: ignore
from gymnasium import spaces

from generals.agents.agent import Agent
from generals.core.game import Action, Game, Info, Observation
from generals.core.grid import GridFactory
from generals.core.replay import Replay
from generals.gui import GUI
from generals.gui.properties import GuiMode

AgentID: TypeAlias = str
Reward: TypeAlias = float
RewardFn: TypeAlias = Callable[[Observation, Action, bool, Info], Reward]


class PettingZooGenerals(pettingzoo.ParallelEnv):
    metadata: dict[str, Any] = {
        "render_modes": ["human"],
        "render_fps": 6,
    }

    def __init__(
        self,
        agents: dict[AgentID, Agent],
        grid_factory: GridFactory | None = None,
        truncation: int | None = None,
        reward_fn: RewardFn | None = None,
        render_mode=None,
    ):
        self.render_mode = render_mode
        self.grid_factory = grid_factory if grid_factory is not None else GridFactory()
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward

        # Agents
        self.agent_data = {agents[id].id: {"color": agents[id].color} for id in agents}
        self.agents = [agents[id].id for id in agents]
        self.possible_agents = self.agents

        assert len(self.possible_agents) == len(
            set(self.possible_agents)
        ), "Agent ids must be unique - you can pass custom ids to agent constructors."
        self.truncation = truncation

    @functools.cache
    def observation_space(self, agent: AgentID) -> spaces.Space:
        assert agent in self.possible_agents, f"Agent {agent} not in possible agents"
        return self.game.observation_space

    @functools.cache
    def action_space(self, agent: AgentID) -> spaces.Space:
        assert agent in self.possible_agents, f"Agent {agent} not in possible agents"
        return self.game.action_space

    def render(self):
        if self.render_mode == "human":
            _ = self.gui.tick(fps=self.metadata["render_fps"])

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, Observation], dict[AgentID, dict]]:
        if options is None:
            options = {}
        self.agents = deepcopy(self.possible_agents)
        if "grid" in options:
            grid = self.grid_factory.grid_from_string(options["grid"])
        else:
            grid = self.grid_factory.grid_from_generator(seed=seed)

        self.game = Game(grid, self.agents)

        if self.render_mode == "human":
            self.gui = GUI(self.game, self.agent_data, GuiMode.TRAIN)

        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                grid=grid,
                agent_data=self.agent_data,
            )
            self.replay.add_state(deepcopy(self.game.channels))
        elif hasattr(self, "replay"):
            del self.replay

        observations = {agent: self.game.agent_observation(agent).as_dict() for agent in self.agents}
        infos: dict[str, Any] = {agent: {} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[AgentID, Action]
    ) -> tuple[
        dict[AgentID, Observation],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, Info],
    ]:
        observations, infos = self.game.step(actions)
        observations = {agent: observation.as_dict() for agent, observation in observations.items()}
        # You probably want to set your truncation based on self.game.time
        truncation = False if self.truncation is None else self.game.time >= self.truncation
        truncated = {agent: truncation for agent in self.agents}
        terminated = {agent: True if self.game.is_done() else False for agent in self.agents}
        rewards = {
            agent: self.reward_fn(
                observations[agent],
                actions[agent],
                terminated[agent] or truncated[agent],
                infos[agent],
            )
            for agent in self.agents
        }

        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))

        # if any agent dies, all agents are terminated
        terminate = any(terminated.values())
        if terminate:
            self.agents = []
            if hasattr(self, "replay"):
                self.replay.store()
        return observations, rewards, terminated, truncated, infos

    @staticmethod
    def _default_reward(
        observation: Observation,
        action: Action,
        done: bool,
        info: Info,
    ) -> Reward:
        return 0

    def close(self) -> None:
        if self.render_mode == "human":
            self.gui.close()
