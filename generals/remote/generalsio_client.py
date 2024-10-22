import numpy as np
from socketio import SimpleClient  # type: ignore

from generals.agents.agent import Agent
from generals.core.config import Observation


class GeneralsBotError(Exception):
    """Base generals-bot exception
    TODO: find a place for exceptions
    """

    pass


class GeneralsIOClientError(GeneralsBotError):
    """Base GeneralsIOClient exception"""

    pass


class RegisterAgentError(GeneralsIOClientError):
    """Registering bot error"""

    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg

    def __str__(self) -> str:
        return f"Failed to register the agent. Error: {self.msg}"


def apply_diff(old: list[int], diff: list[int]) -> list[int]:
    i = 0
    new = []
    while i < len(diff):
        if diff[i] > 0:  # matching
            new.extend(old[len(new) : len(new) + diff[i]])
        i += 1
        if i < len(diff) and diff[i] > 0:  # applying diffs
            new.extend(diff[i + 1 : i + 1 + diff[i]])
            i += diff[i]
        i += 1
    return new

test_old_1 = [0, 0]
test_diff_1 = [1, 1, 3]
desired = [0,3]
assert apply_diff(test_old_1, test_diff_1) == desired
test_old_2 = [0,0]
test_diff_2 = [0,1,2,1]
desired = [2, 0]
assert apply_diff(test_old_2, test_diff_2) == desired
print("All tests passed")


class GeneralsIOState:
    def __init__(self, data: dict):
        self.replay_id = data["replay_id"]
        self.usernames = data["usernames"]
        self.player_index = data["playerIndex"]
        self.opponent_index = 1 - self.player_index # works only for 1v1

        self.n_players = len(self.usernames)

        self.map = []
        self.cities = []

    def update(self, data: dict) -> None:
        self.turn = data["turn"]
        self.map = apply_diff(self.map, data["map_diff"])
        self.cities = apply_diff(self.cities, data["cities_diff"])
        self.generals = data["generals"]
        self.scores = data["scores"]
        if "stars" in data:
            self.stars = data["stars"]


    def agent_observation(self) -> Observation:
        width, height = self.map[0], self.map[1]
        size = height * width

        armies = np.array(self.map[2 : 2 + size]).reshape((height, width))
        terrain = np.array(self.map[2 + size : 2 + 2 * size]).reshape((height, width))

        # make 2D binary map of owned cells. These are the ones that have self.player_index value in terrain
        army = armies
        owned_cells = np.where(terrain == self.player_index, 1, 0)
        opponent_cells = np.where(terrain == self.opponent_index, 1, 0)
        visible_neutral_cells = np.where(terrain == -1, 1, 0)
        print(self.generals)


class GeneralsIOClient(SimpleClient):
    """
    Wrapper around socket.io client to enable Agent to join
    GeneralsIO lobby.
    """

    def __init__(self, agent: Agent, user_id: str):
        super().__init__()
        self.connect("https://botws.generals.io")
        self.user_id = user_id
        self._queue_id = ""

    @property
    def queue_id(self):
        if not self._queue_id:
            raise GeneralsIOClientError("Queue ID is not set.\nIs agent in the game lobby?")

        return self._queue_id

    def _emit_receive(self, *args):
        self.emit(*args)
        return self.receive()

    def register_agent(self, username: str) -> None:
        """
        Register Agent to GeneralsIO platform.
        You can configure one Agent per `user_id`. `user_id` should be handled as secret.
        :param user_id: secret ID of Agent
        :param username: agent username, must be prefixed with `[Bot]`
        """
        event, response = self._emit_receive("set_username", (self.user_id, username))
        if response:
            # in case of success the response is empty
            raise RegisterAgentError(response)

    def join_private_lobby(self, queue_id: str) -> None:
        """
        Join (or create) private game lobby.
        :param queue_id: Either URL or lobby ID number
        """
        self._emit_receive("join_private", (queue_id, self.user_id))
        self._queue_id = queue_id

    def join_game(self, force_start: bool = True) -> None:
        """
        Set force start if requested and wait for the game start.
        :param force_start: If set to True, the Agent will set `Force Start` flag
        """
        if force_start:
            self.emit("set_force_start", (self.queue_id, True))

        while True:
            event, *data = self.receive()
            if event == "game_start":
                self._initialize_game(data)
                break

        self._play_game()

    def _initialize_game(self, data: dict) -> None:
        """
        Triggered after server starts the game.
        :param data: dictionary of information received in the beginning
        """
        self.game_state = GeneralsIOState(data[0])

    def _play_game(self) -> None:
        """
        Triggered after server starts the game.
        TODO: spawn a new thread in which Agent will calculate its moves
        """
        winner = False
        # TODO deserts?
        while True:
            event, data, suffix = self.receive()
            print("received an event:", event)
            match event:
                case "game_update":
                    self.game_state.update(data)
                    self.game_state.agent_observation()
                case "game_lost" | "game_won":
                    # server sends game_lost or game_won before game_over
                    winner = event == "game_won"
                    break

        self._finish_game(winner)

    def _finish_game(self, is_winner: bool) -> None:
        """
        Triggered after server finishes the game.
        :param is_winner: True if Agent won the game
        """
        print("game is finished. Am I a winner?", is_winner)
