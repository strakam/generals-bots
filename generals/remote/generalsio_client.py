import numpy as np
from socketio import SimpleClient  # type: ignore

from generals.agents import Agent, AgentFactory
from generals.core.config import Direction
from generals.core.observation import Observation
from generals.remote.generalsio_state import GeneralsIOstate

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


def autopilot(agent_id: str, user_id: str, lobby_id: str) -> None:
    """
    Start the autopilot for the GeneralsIO client.
    This means that agent will join the lobby and force starts,
    so he plays indefinitely.
    """
    agent = AgentFactory.make_agent(agent_id)
    with GeneralsIOClient(agent, user_id) as client:
        while True:
            if client.status == "off":
                client.join_private_lobby(lobby_id)
            if client.status == "lobby":
                client.join_game()


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
        self.agent = agent
        self._queue_id = ""
        self._replay_id = ""
        self._status = "off"  # can be "off","game","lobby","queue"

    @property
    def queue_id(self):
        if not self._queue_id:
            raise GeneralsIOClientError("Queue ID is not set.\nIs agent in the game lobby?")

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
        event, response = self._emit_receive("set_username", (self.user_id, username))
        if response:
            # in case of success the response is empty
            raise RegisterAgentError(response)

    def join_private_lobby(self, queue_id: str) -> None:
        """
        Join (or create) private game lobby.
        :param queue_id: Either URL or lobby ID number
        """
        self._status = "lobby"
        self._emit_receive("join_private", (queue_id, self.user_id))
        self._queue_id = queue_id

    def join_game(self, force_start: bool = True) -> None:
        """
        Set force start if requested and wait for the game start.
        :param force_start: If set to True, the Agent will set `Force Start` flag
        """
        self._status = "queue"
        while True:
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
        self._emit_receive("join_1v1", self.user_id)
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

    def _generate_action(self, observation: Observation) -> tuple[int, int, int] | None:
        """
        Translate action from Agent to the server format.
        :param action: dictionary representing the action
        """
        obs = observation.as_dict()
        action = self.agent.act(obs)
        if not action["pass"]:
            source: np.ndarray = np.array(action["cell"])
            direction = np.array(DIRECTIONS[action["direction"]].value)
            split = action["split"]
            destination = source + direction
            # convert to index
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
            event, data, _ = self.receive()
            match event:
                case "game_update":
                    self.game_state.update(data)
                    obs = self.game_state.get_observation()
                    action = self._generate_action(obs)
                    if action:
                        self.emit("attack", action)
                case "game_lost" | "game_won":
                    self._finish_game(event == "game_won")
                    break

    def _finish_game(self, is_winner: bool) -> None:
        """
        Triggered after server finishes the game.
        :param is_winner: True if Agent won the game
        """
        self._status = "off"
        status = "won!" if is_winner else "lost."
        print(f"Game is finished, you {status}, replay ID: https://bot.generals.io/replays/{self.replay_id}")
        self.emit("leave_game")
