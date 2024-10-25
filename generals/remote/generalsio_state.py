import numpy as np

from generals.core.observation import Observation


class GeneralsIOstate:
    def __init__(self, data: dict):
        self.usernames = data["usernames"]
        self.player_index = data["playerIndex"]
        self.opponent_index = 1 - self.player_index  # works only for 1v1

        self.n_players = len(self.usernames)

        self.map: list[int] = []
        self.cities: list[int] = []

    def update(self, data: dict) -> None:
        self.turn = data["turn"]
        self.map = self.apply_diff(self.map, data["map_diff"])
        self.cities = self.apply_diff(self.cities, data["cities_diff"])
        self.generals = data["generals"]
        self.scores = data["scores"]
        if "stars" in data:
            self.stars = data["stars"]

    def apply_diff(self, old: list[int], diff: list[int]) -> list[int]:
        i = 0
        new: list[int] = []
        while i < len(diff):
            if diff[i] > 0:  # matching
                new.extend(old[len(new) : len(new) + diff[i]])
            i += 1
            if i < len(diff) and diff[i] > 0:  # applying diffs
                new.extend(diff[i + 1 : i + 1 + diff[i]])
                i += diff[i]
            i += 1
        return new

    def get_observation(self) -> Observation:
        width, height = self.map[0], self.map[1]
        size = height * width

        armies = np.array(self.map[2 : 2 + size]).reshape((height, width))
        terrain = np.array(self.map[2 + size : 2 + 2 * size]).reshape((height, width))
        cities = np.zeros((height, width))
        for city in self.cities:
            cities[city // width, city % width] = 1

        generals = np.zeros((height, width))
        for general in self.generals:
            if general != -1:
                generals[general // width, general % width] = 1

        army = armies
        owned_cells = np.where(terrain == self.player_index, 1, 0)
        opponent_cells = np.where(terrain == self.opponent_index, 1, 0)
        neutral_cells = np.where(terrain == -1, 1, 0)
        mountain_cells = np.where(terrain == -2, 1, 0)
        fog_cells = np.where(terrain == -3, 1, 0)
        structures_in_fog = np.where(terrain == -4, 1, 0)
        owned_land_count = self.scores[self.player_index]["tiles"]
        owned_army_count = self.scores[self.player_index]["total"]
        opponent_land_count = self.scores[self.opponent_index]["tiles"]
        opponent_army_count = self.scores[self.opponent_index]["total"]
        timestep = self.turn

        return Observation(
            armies=army,
            generals=generals,
            cities=cities,
            mountains=mountain_cells,
            neutral_cells=neutral_cells,
            owned_cells=owned_cells,
            opponent_cells=opponent_cells,
            fog_cells=fog_cells,
            structures_in_fog=structures_in_fog,
            owned_land_count=owned_land_count,
            owned_army_count=owned_army_count,
            opponent_land_count=opponent_land_count,
            opponent_army_count=opponent_army_count,
            timestep=timestep,
        )
