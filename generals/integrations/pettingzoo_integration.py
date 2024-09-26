import functools
import pettingzoo
from copy import deepcopy
from ..game import Game
from ..replay import Replay
from ..rendering import Renderer
from collections import OrderedDict


class PZ_Generals(pettingzoo.ParallelEnv):
    def __init__(self, mapper=None, agents=None, reward_fn=None, render_mode=None):
        self.render_mode = render_mode
        self.mapper = mapper

        self.agent_data = {
            agent.name: {"color": agent.color} for agent in agents
        }
        self.possible_agents = [agent.name for agent in agents]

        assert (
            len(self.possible_agents) == len(set(self.possible_agents))
        ), "Agent names must be unique - you can pass custom names to agent constructors."

        self.reward_fn = self.default_rewards if reward_fn is None else reward_fn

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name):
        assert agent_name in self.agents, f"{agent_name} is not a valid agent"
        return self.game.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_name):
        assert agent_name in self.agents, f"{agent_name} is not a valid agent"
        return self.game.action_space

    def render(self, tick_rate=None):
        if self.render_mode == "human":
            self.renderer.render()
            if tick_rate is not None:
                self.renderer.clock.tick(tick_rate)

    def reset(self, seed=None, options={}):
        self.agents = deepcopy(self.possible_agents)

        map = self.mapper.get_map(numpify=True)
        self.game = Game(map, self.possible_agents)

        if self.render_mode == "human":
            from_replay = "from_replay" in options and options["from_replay"] or False
            self.renderer = Renderer(self.game, self.agent_data, from_replay)

        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                map=self.mapper.get_map(numpify=False),
                agent_data=self.agent_data,
            )
            self.replay.add_state(deepcopy(self.game.channels))

        observations = OrderedDict(
            {agent: self.game._agent_observation(agent) for agent in self.agents}
        )
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action):
        observations, infos = self.game.step(action)

        truncated = {agent: False for agent in self.agents}  # no truncation
        terminated = {
            agent: True if self.game.is_done() else False for agent in self.agents
        }
        rewards = self.reward_fn(observations)

        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))

        # if any agent dies, all agents are terminated
        terminate = any(terminated.values())
        if terminate:
            self.agents = []
            if hasattr(self, "replay"):
                self.replay.store()

        return OrderedDict(observations), rewards, terminated, truncated, infos

    def default_rewards(self, observations):
        """
        Calculate rewards for each agent.
        Give 0 if game still running, otherwise 1 for winner and -1 for loser.
        """
        rewards = {agent: 0 for agent in self.agents}
        game_ended = any(observations[agent]["is_winner"] for agent in self.agents)
        if game_ended:
            for agent in self.agents:
                if observations[agent]["is_winner"]:
                    rewards[agent] = 1
                else:
                    rewards[agent] = -1
        return rewards

    def close(self):
        print("Closing environment")
