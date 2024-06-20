import functools
from copy import copy
import time


import pettingzoo

from typing import List

from . import game, config, utils, map


def generals_v0(config=config.Config, render_mode="human"):
    """
    Here we apply wrappers to the environment.
    """
    env = Generals(config, render_mode=render_mode)
    return env


class Generals(pettingzoo.ParallelEnv):
    metadata = {"render.modes": ["human", "none"]}

    def __init__(
        self,
        game_config: config.Config,
        agent_names: List[str] = ["red", "blue"],
        render_mode="human",
    ):
        self.game_config = game_config
        self.render_mode = render_mode

        self.agents = agent_names
        self.possible_agents = self.agents[:]

        self.name_to_id = dict(zip(agent_names, list(range(1, len(agent_names) + 1))))

        if render_mode == "human":
            utils.init_screen(self.game_config)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name):
        assert agent_name in self.agents, f"{agent_name} is not a valid agent"
        return self.game.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_name):
        assert agent_name in self.agents, f"{agent_name} is not a valid agent"
        return self.game.action_space

    def render(self):
        if self.render_mode == "human":
            utils.handle_events()
            utils.render_grid(self.game, self.agents)
            utils.render_gui(self.game, self.agents)
            utils.pygame.display.flip()
            time.sleep(0.1)  # move to constants

    def reset(self, seed=None, options={}):
        self.agents = copy(self.possible_agents)
        if "map" in options:
            _map = options["map"]
        else:
            _map = map.generate_map(
                self.game_config.grid_size,
                self.game_config.town_density,
                self.game_config.mountain_density,
                len(self.agents),
            )

        self.game = game.Game(_map, self.possible_agents)

        if "replay" in options and options["replay"]:
            self.replay = True
            self.action_history = []

        observations = {
            agent: self.game._agent_observation(agent) for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.game.step(actions)

        if self.replay:
            self.action_history.append(actions)

        # if any agent dies, all agents are terminated
        terminate = any(terminated.values())
        if terminate:
            terminated = {agent: True for agent in self.agents}
            self.agents = []
            # if replay is on, store the game
            if self.replay:
                map.store_replay(
                    self.game.map, self.action_history, self.possible_agents
                )

        return observations, rewards, terminated, truncated, infos

    def close(self):
        utils.pygame.quit()
