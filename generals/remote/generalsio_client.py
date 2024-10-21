import numpy as np
from socketio import SimpleClient  # type: ignore

from generals.agents.agent import Agent


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

        agent_index = None
        while True:
            event, *data = self.receive()
            if event == "game_start":
                game_data = data[0]
                agent_index = game_data["playerIndex"]
                break

        self._play_game(agent_index)

    def _play_game(self, agent_index: int) -> None:
        """
        Triggered after server starts the game.
        TODO: spawn a new thread in which Agent will calculate its moves
        :param agent_index: The index of agent in the game
        """
        winner = False
        map = np.empty([])  # noqa: F841
        cities = np.empty([])  # noqa: F841
        # TODO deserts?
        while True:
            event, data, suffix = self.receive()
            print('received an event:', event)
            match event:
                case "game_update":
                    map_diff = np.array(data["map_diff"])  # noqa: F841
                    cities_diff = np.array(data["cities_diff"])  # noqa: F841
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
