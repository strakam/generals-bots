import pickle
import time
from generals.map import Mapper
from generals.rendering import Renderer
from generals.game import Game
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
        path = self.name if self.name.endswith(".pkl") else self.name + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Replay successfully stored as {path}")

    @classmethod
    def load(cls, path):
        path = path if path.endswith(".pkl") else path + ".pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def play(self):
        map = Mapper.numpify_map(self.map)
        agents = [agent for agent in self.agent_data.keys()]
        game = Game(map, agents)
        renderer = Renderer(game, self.agent_data, from_replay=True)

        game_step, last_input_time, last_move_time = 0, 0, 0
        while 1:
            _t = time.time()
            # Check inputs
            if _t - last_input_time > 0.008:  # check for input every 8ms
                control_events = renderer.render()
                last_input_time = _t
            else:
                control_events = {"time_change": 0}
            if "restart" in control_events:
                game_step = 0
            # If we control replay, change game state
            game_step = max(
                0, min(len(self.game_states) - 1, game_step + control_events["time_change"])
            )
            if renderer.paused and game_step != game.time:
                game.channels = deepcopy(self.game_states[game_step])
                game.time = game_step
                last_move_time = _t
            # If we are not paused, play the game
            elif (
                _t - last_move_time > renderer.game_speed * 0.512
                and not renderer.paused
            ):
                if game.is_done():
                    renderer.paused = True
                game_step = min(len(self.game_states) - 1, game_step + 1)
                game.channels = deepcopy(self.game_states[game_step])
                game.time = game_step
                last_move_time = _t
            renderer.clock.tick(60)
