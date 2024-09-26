import pickle
import time
from generals.agent import RandomAgent
from generals.map import Mapper
from copy import deepcopy


class Replay:
    def __init__(self, name, map, agent_data):
        self.name = name
        self.map = map
        self.agent_data = agent_data

        self.game_states = []

    def add_state(self, state):
        self.game_states.append(state)

    def store(self):
        # if self.name is not suffixed with .pkl, add it
        path = self.name if self.name.endswith(".pkl") else self.name + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        path = path if path.endswith(".pkl") else path + ".pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def play(self):
        from generals.env import pz_generals
        agents = [
            RandomAgent(name=agent, color=self.agent_data[agent]["color"])
            for agent in self.agent_data.keys()
        ]
        mapper = Mapper(grid_size=16, mountain_density=0.2, city_density=0.05)
        env = pz_generals(mapper=mapper, agents=agents, render_mode="human")
        mapper.map = self.map
        env.reset(self.map, options={"from_replay": True})
        env.renderer.render()

        game_step, last_input_time, last_move_time = 0, 0, 0
        while 1:
            _t = time.time()
            # Check inputs
            if _t - last_input_time > 0.008:  # check for input every 8ms
                control_events = env.renderer.render()
                last_input_time = _t
            else:
                control_events = {"time_change": 0}
            # If we control replay, change game state
            game_step = max(
                0, min(len(self.game_states) - 1, game_step + control_events["time_change"])
            )
            if env.renderer.paused and game_step != env.game.time:
                env.agents = deepcopy(env.possible_agents)
                env.game.channels = deepcopy(self.game_states[game_step])
                env.game.time = game_step
                last_move_time = _t
            # If we are not paused, play the game
            elif (
                _t - last_move_time > env.renderer.game_speed * 0.512
                and not env.renderer.paused
            ):
                if env.game.is_done():
                    env.renderer.paused = True
                game_step = min(len(self.game_states) - 1, game_step + 1)
                env.game.channels = deepcopy(self.game_states[game_step])
                env.game.time = game_step
                last_move_time = _t
            env.renderer.clock.tick(60)
