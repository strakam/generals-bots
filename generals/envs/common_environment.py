from collections.abc import Callable
from typing import TypeAlias

from copy import deepcopy
import gymnasium as gym

from generals.core.game import Game, Action, Observation, Info
from generals.core.grid import GridFactory
from generals.core.replay import Replay
from generals.gui import GUI
from generals.gui.properties import GuiMode


# Type aliases
Reward: TypeAlias = float
RewardFn: TypeAlias = Callable[[dict[str, Observation], Action, bool, Info], Reward]
AgentID: TypeAlias = str


class CommonEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 6,
    }
    default_colors = [
        (67, 70, 86),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]  # Up for improvement (needs to be extended for multiple agents)

    def __init__(
        self,
        grid_factory: GridFactory,
        render_mode=None,
    ):
        self.render_mode = render_mode
        self.grid_factory = grid_factory
        self.reward_fn = self._default_reward

    @property
    def reward_fn(self) -> RewardFn:
        return self._reward_fn

    @reward_fn.setter
    def reward_fn(self, rew_fn: RewardFn):
        self._reward_fn = rew_fn

    def _render(self, fps: int = None) -> None:
        fps = self.metadata["render_fps"] if fps is None else fps
        if self.render_mode == "human":
            _ = self.gui.tick(fps=fps)

    def _reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, Observation], dict[AgentID, dict]]:
        super().reset(seed=seed)
        if "grid" in options:
            grid = self.grid_factory.grid_from_string(options["grid"])
        else:
            grid = self.grid_factory.grid_from_generator(seed=seed)

        self.game = Game(grid, self.agent_ids)

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

        observations = self.game.get_all_observations()
        infos = {agent: {} for agent in self.agent_ids}
        return observations, infos

    def _step(
        self, actions: dict[AgentID, Action]
    ) -> tuple[
        dict[AgentID, Observation],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, Info],
    ]:
        observations, infos = self.game.step(actions)

        truncated = {agent: False for agent in self.agent_ids}  # no truncation
        terminated = {
            agent: True if self.game.is_done() else False for agent in self.agent_ids
        }

        rewards = {
            agent: self.reward_fn(
                observations[agent],
                actions[agent],
                terminated[agent] or truncated[agent],
                infos[agent],
            )
            for agent in self.agent_ids
        }

        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))

        # if any agent dies, all agents are terminated
        terminate = any(terminated.values())
        if terminate:
            self.agents_ids = []
            if hasattr(self, "replay"):
                self.replay.store()

        return observations, rewards, terminated, truncated, infos

    @staticmethod
    def _default_reward(
        observation: dict[str, Observation],
        action: Action,
        done: bool,
        info: Info,
    ) -> Reward:
        """
        Give 0 if game still running, otherwise 1 for winner and -1 for loser.
        """
        if done:
            reward = 1 if observation["observation"]["is_winner"] else -1
        else:
            reward = 0
        return reward

    def close(self) -> None:
        self.gui.close()