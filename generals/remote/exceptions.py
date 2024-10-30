from generals import GeneralsBotError


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
