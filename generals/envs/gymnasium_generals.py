from collections.abc import Callable
from copy import deepcopy
from typing import Any, SupportsFloat, TypeAlias

import gymnasium as gym

from generals.agents import Agent, AgentFactory
from generals.core.game import Action, Game, Info
from generals.core.grid import GridFactory
from generals.core.observation import Observation
from generals.core.replay import Replay
from generals.gui import GUI
from generals.gui.properties import GuiMode

Reward: TypeAlias = float
RewardFn: TypeAlias = Callable[[Observation, Action, bool, Info], Reward]


class GymnasiumGenerals(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 6,
    }

    def __init__(
        self,
        grid_factory: GridFactory | None = None,
        npc: Agent | None = None,
        agent: Agent | None = None,  # Optional, just to obtain id and color
        truncation: int | None = None,
        reward_fn: RewardFn | None = None,
        render_mode: str | None = None,
    ):
        self.render_mode = render_mode
        self.grid_factory = grid_factory if grid_factory is not None else GridFactory()
        self.reward_fn = reward_fn if reward_fn is not None else GymnasiumGenerals._default_reward

        # Agents
        if npc is None:
            print('No NPC agent provided. Creating "Random" NPC as a fallback.')
            npc = AgentFactory.make_agent("random")
        else:
            assert isinstance(npc, Agent), "NPC must be an instance of Agent class."
        self.npc = npc
        self.agent_id = "Agent" if agent is None else agent.id
        self.agent_ids = [self.agent_id, self.npc.id]
        self.agent_data = {
            self.agent_id: {"color": (67, 70, 86) if agent is None else agent.color},
            self.npc.id: {"color": self.npc.color},
        }
        assert self.agent_id != npc.id, "Agent ids must be unique - you can pass custom ids to agent constructors."

        # Game
        grid = self.grid_factory.grid_from_generator()
        self.game = Game(grid, [self.agent_id, self.npc.id])
        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space
        self.truncation = truncation

    def render(self):
        if self.render_mode == "human":
            _ = self.gui.tick(fps=self.metadata["render_fps"])

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        if options is None:
            options = {}

        if "grid" in options:
            grid = self.grid_factory.grid_from_string(options["grid"])
        else:
            self.grid_factory.rng = self.np_random
            grid = self.grid_factory.grid_from_generator()

        # Create game for current run
        self.game = Game(grid, self.agent_ids)

        # Create GUI for current render run
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

        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space

        observation = self.game.agent_observation(self.agent_id).as_dict()
        info: dict[str, Any] = {}
        return observation, info

    def step(self, action: Action) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        # Get action of NPC
        npc_observation = self.game.agent_observation(self.npc.id).as_dict()
        npc_action = self.npc.act(npc_observation)
        actions = {self.agent_id: action, self.npc.id: npc_action}

        observations, infos = self.game.step(actions)
        infos = {agent_id: {} for agent_id in self.agent_ids}

        # From observations of all agents, pick only those relevant for the main agent
        obs = observations[self.agent_id].as_dict()
        info = infos[self.agent_id]
        reward = self.reward_fn(obs, action, self.game.is_done(), info)
        terminated = self.game.is_done()
        truncated = False
        if self.truncation is not None:
            truncated = self.game.time >= self.truncation

        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))

        if terminated or truncated:
            if hasattr(self, "replay"):
                self.replay.store()
        return obs, reward, terminated, truncated, info

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
