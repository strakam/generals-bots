import time

import numpy as np
from socketio import SimpleClient  # type: ignore

from generals.agents import Agent
from generals.core.config import Direction
from generals.core.observation import Observation
from generals.remote.generalsio_state import GeneralsIOstate

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

PUBLIC_ENDPOINT = "https://ws.generals.io/"
BOT_ENDPOINT = "https://botws.generals.io/"


def autopilot(agent: Agent, user_id: str, lobby_id: str) -> None:
    """
    Start the autopilot for the GeneralsIO client.
    This means that agent will join the lobby and force starts,
    so he plays indefinitely.
    """
    with GeneralsIOClient(agent, user_id) as client:
        while True:
            if client.status == "off":
                client.join_private_lobby(lobby_id)
            if client.status == "lobby":
                client.join_game()


class GeneralsIOClient(SimpleClient):
    """
    Wrapper around socket.io client to enable Agent to join
    GeneralsIO lobby.
    """

    def __init__(self, agent: Agent, user_id: str, public_server: bool = False):
        super().__init__()
        self.public_server = public_server
        self.user_id = user_id
        self.agent = agent
        self._queue_id = ""
        self._replay_id = ""
        self._status = "off"  # can be "off","game","lobby","queue"
        self.bot_key = "sd09fjd203i0ejwi_changeme"

        self.connect(PUBLIC_ENDPOINT if public_server else BOT_ENDPOINT)
        print("Connected to server!")

    @property
    def queue_id(self):
        if not self._queue_id:
            raise ValueError("No queue ID available.")

        return self._queue_id

    @property
    def replay_id(self):
        if not self._replay_id:
            print("No replay ID available.")
        return self._replay_id

    @property
    def status(self):
        return self._status

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
        payload = (self.user_id, username, self.bot_key)
        _, response = self._emit_receive("set_username", payload)
        if response:  # in case of success the response is empty
            raise ValueError(f"Failed to register the agent: {response}.")
        print(f"Agent {username} registered!")

    def join_private_lobby(self, lobby_id: str) -> None:
        """
        Join (or create) private game lobby.
        :param queue_id: Either URL or lobby ID number
        """
        self._status = "lobby"
        payload = (lobby_id, self.user_id, self.bot_key)
        self._emit_receive("join_private", payload)
        self._queue_id = lobby_id
        print(f"Joined private lobby {lobby_id}.")

    def join_game(self, force_start: bool = True) -> None:
        """
        Set force start if requested and wait for the game start.
        :param force_start: If set to True, the Agent will set `Force Start` flag
        """
        self._status = "queue"
        while True:
            time.sleep(2)  # dont spam servers
            event, *data = self.receive()
            self.emit("set_force_start", (self.queue_id, force_start))
            if event == "game_start":
                self._status = "game"
                self._initialize_game(data)
                self._play_game()
                break

    def join_1v1_queue(self) -> None:
        """
        Join 1v1 queue.
        """
        self._status = "queue"
        payload = (self.user_id, self.bot_key)
        self._emit_receive("join_1v1", payload)
        while True:
            event, *data = self.receive()
            if event == "game_start":
                self._status = "game"
                self._initialize_game(data)
                self._play_game()
                break

    def _initialize_game(self, data: dict) -> None:
        """
        Triggered after server starts the game.
        :param data: dictionary of information received in the beginning
        """
        self.game_state = GeneralsIOstate(data[0])
        self._replay_id = data[0]["replay_id"]
        print("Game started!")

    def _generate_action(self, observation: Observation) -> tuple[int, int, int] | None:
        """
        Translate action from Agent to the server format.
        :param action: dictionary representing the action
        If your agent passes actions correctly into our simulator, it will work here too.
        """
        action = self.agent.act(observation)
        pass_or_play = action[0]
        i, j = action[1], action[2]
        direction = action[3]
        split = action[4]
        if not pass_or_play:
            source: np.ndarray = np.array([i, j])
            direction = np.array(DIRECTIONS[direction].value)
            destination = source + direction
            source_index = source[0] * self.game_state.map[0] + source[1]
            destination_index = destination[0] * self.game_state.map[0] + destination[1]
            return (int(source_index), int(destination_index), int(split))
        return None

    def _play_game(self) -> None:
        """
        Main game-play loop.
        TODO: spawn a new thread in which Agent will calculate its moves
        """
        while True:
            try:
                event, data, _ = self.receive()
            except ValueError:
                self._finish_game(is_winner=True)
                return
            match event:
                case "game_update":
                    self.game_state.update(data)
                    obs = self.game_state.get_observation()
                    action = self._generate_action(obs)
                    if action:
                        self.emit("attack", action)
                case "game_lost" | "game_won":
                    self._finish_game(event == "game_won")
                    return

    def _finish_game(self, is_winner: bool) -> None:
        """
        Triggered after server finishes the game.
        :param is_winner: True if Agent won the game
        """
        self._status = "off"
        status = "Won!" if is_winner else "Lost."
        prefix = "bot." if not self.public_server else ""
        print(f"You {status}, replay link: https://{prefix}generals.io/replays/{self.replay_id}")
        self.emit("leave_game")
