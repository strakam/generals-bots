# Optional game modifiers. Each modifier lives in its own module and wraps the
# base game as a pre-step transform — generals.core.game is never modified.
from generals.modifiers import build_castles, deathtouch

__all__ = ["build_castles", "deathtouch"]
