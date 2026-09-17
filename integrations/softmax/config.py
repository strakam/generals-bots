"""Strict runtime config and generated manifest schema."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlayerName(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    name: str = Field(min_length=1, max_length=100)


class GameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    tokens: list[Annotated[str, Field(min_length=1, max_length=1024)]] = Field(min_length=2, max_length=4)
    players: list[PlayerName] = Field(min_length=2, max_length=4)
    ruleset: Literal["classic", "build_castles"] = "classic"
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    max_turns: int = Field(default=1200, ge=1, le=1200)
    turn_timeout_seconds: float = Field(default=0.5, ge=0.05, le=1)
    tick_interval_seconds: float = Field(default=0, ge=0, le=0.5)
    player_connect_timeout_seconds: float = Field(default=180, ge=1, le=180)
    max_consecutive_timeouts: int = Field(default=20, ge=1, le=1200)

    @model_validator(mode="after")
    def unique_tokens(self):
        if len(self.tokens) != len(self.players) or len(self.tokens) not in (2, 4):
            raise ValueError("matching rosters of two or four players are required")
        if self.ruleset == "build_castles" and len(self.players) != 2:
            raise ValueError("castle-building requires two players")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("player tokens must be distinct")
        return self
