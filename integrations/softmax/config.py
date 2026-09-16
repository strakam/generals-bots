"""Strict runtime config and generated manifest schema."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlayerName(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    name: str = Field(min_length=1, max_length=100)


class GameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    tokens: list[Annotated[str, Field(min_length=1, max_length=1024)]] = Field(min_length=2, max_length=2)
    players: list[PlayerName] = Field(min_length=2, max_length=2)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    max_turns: int = Field(default=1200, ge=1, le=2000)
    turn_timeout_seconds: float = Field(default=0.5, ge=0.05, le=1)
    tick_interval_seconds: float = Field(default=0, ge=0, le=0.5)
    player_connect_timeout_seconds: float = Field(default=180, ge=1, le=180)
    max_consecutive_timeouts: int = Field(default=20, ge=1, le=1200)

    @model_validator(mode="after")
    def unique_tokens(self):
        if self.tokens[0] == self.tokens[1]:
            raise ValueError("player tokens must be distinct")
        return self
