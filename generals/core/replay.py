import pickle
import time
from copy import deepcopy
from typing import Any

from generals.core.channels import Channels
from generals.core.game import Game
from generals.core.grid import Grid
from generals.gui import GUI
from generals.gui.event_handler import ReplayCommand
from generals.gui.properties import GuiMode


class Replay:
    def __init__(self, name: str, grid: Grid, agent_data: dict[str, Any]):
        self.name = name
        self.grid = grid
        self.agent_data = agent_data

        self.game_states: list[Channels] = []

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
        agents = [agent for agent in self.agent_data.keys()]
        game = Game(self.grid, agents)
        gui = GUI(game, self.agent_data, mode=GuiMode.REPLAY)
        gui_properties = gui.properties

        game_step, last_input_time, last_move_time = 0, 0, 0
        while 1:
            _t = time.time()
            # Check inputs
            if _t - last_input_time > 0.008:  # check for input every 8ms
                command = gui.tick()
                last_input_time = _t
            else:
                command = ReplayCommand()
            if command.restart:
                game_step = 0
            # If we control replay, change game state
            game_step = max(0, min(len(self.game_states) - 1, game_step + command.frame_change))
            if gui_properties.paused and game_step != game.time:
                game.channels = deepcopy(self.game_states[game_step])
                game.time = game_step
                last_move_time = _t
            # If we are not paused, play the game
            elif _t - last_move_time > (1 / gui_properties.game_speed) * 0.512 and not gui_properties.paused:
                if game.is_done():
                    gui_properties.paused = True
                game_step = min(len(self.game_states) - 1, game_step + 1)
                game.channels = deepcopy(self.game_states[game_step])
                game.time = game_step
                last_move_time = _t
            gui_properties.clock.tick(60)
