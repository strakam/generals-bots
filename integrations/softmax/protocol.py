"""Versioned JSON wire contract. No engine or server dependencies."""

VERSION = 1
PASS = [1, 0, 0, 0, 0]


def parse_action(message: object, turn: int, height: int, width: int, ruleset: str = "classic") -> list[int]:
    """Reject malformed, stale, or unbounded inputs before touching JAX."""
    if not isinstance(message, dict) or set(message) != {"type", "turn", "action"}:
        raise ValueError("expected type, turn and action")
    if message["type"] != "action" or type(message["turn"]) is not int or message["turn"] != turn:
        raise ValueError("action must address the current turn")
    action = message["action"]
    if not isinstance(action, list) or len(action) != 5 or any(type(v) is not int for v in action):
        raise ValueError("action must contain five integers")
    kind, row, col, direction, split = action
    allowed = (0, 1, 2) if ruleset == "build_castles" else (0, 1)
    if kind not in allowed or not (0 <= row < height and 0 <= col < width):
        raise ValueError("invalid action kind or source cell")
    if direction not in (0, 1, 2, 3) or split not in (0, 1):
        raise ValueError("invalid direction or split")
    return action


def stdio_frame(observation: dict) -> str:
    """Translate a JSON observation into the existing competition protocol."""
    o = observation
    lines = [f"{o['turn']} {o['my_land']} {o['my_army']} {o['opp_land']} {o['opp_army']}"]
    for name in ("type_grid", "owner_grid", "army_grid"):
        lines.extend(" ".join(map(str, row)) for row in o[name])
    return "\n".join(lines) + "\n"
