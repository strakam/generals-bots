"""Generate the Coworld template from runtime schema and game-owned docs."""

import argparse
import json
from pathlib import Path

from integrations.softmax.config import GameConfig

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/strakam/generals-bots/tree/softmax/integrations/softmax"


def document(filename):
    return {"type": "text", "value": (ROOT / filename).read_text()}


def template():
    scores = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
    counts = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0}}
    results = {
        "type": "object",
        "additionalProperties": False,
        "required": ["scores", "winner", "reason", "turns", "army", "land", "timeouts"],
        "properties": {
            "scores": scores,
            "winner": {"type": "integer", "enum": [-1, 0, 1]},
            "reason": {"type": "string", "enum": ["general_capture", "turn_limit", "forfeit", "double_forfeit"]},
            "turns": {"type": "integer", "minimum": 0, "maximum": 1200},
            "army": counts,
            "land": counts,
            "timeouts": counts,
        },
    }
    players = [{"name": "Red"}, {"name": "Blue"}]
    return {
        "$schema": "https://raw.githubusercontent.com/Metta-AI/coworld/4c26e51/src/coworld/coworld_manifest_schema.json",
        "tags": ["strategy", "1v1", "fog-of-war", "territory-control"],
        "game": {
            "name": "generals-competition",
            "owner": "Matej Straka",
            "description": (
                "Capture the enemy general in a seeded 1v1 fog-of-war strategy game. "
                "Capture neutral castles. Regular combat; no castle building or Deathtouch."
            ),
            "runnable": {
                "type": "game",
                "image": "{{GENERALS_GAME_IMAGE}}",
                "run": ["python", "-m", "integrations.softmax.server"],
                "source_url": SOURCE,
            },
            "config_schema": GameConfig.model_json_schema(),
            "results_schema": results,
            "protocols": {"player": document("PLAYER_PROTOCOL.md"), "global": document("GLOBAL_PROTOCOL.md")},
            "docs": {"readme": document("README.md")},
            "replay_viewer": {"bundle": "replay-viewer", "replay_compression": "gzip"},
        },
        "player": [
            {
                "id": "expander",
                "name": "Expander",
                "description": "Pure-Python competition baseline through the stdio bridge.",
                "type": "player",
                "image": "{{GENERALS_PLAYER_IMAGE}}",
                "run": ["python", "-m", "integrations.softmax.player"],
                "source_url": SOURCE,
            }
        ],
        "variants": [
            {
                "id": "competition",
                "name": "Classic 1v1",
                "description": "Regular capture-only rules, 1200-turn cap, fresh private seed; fast bot play.",
                "game_config": {"players": players, "max_turns": 1200},
            },
            {
                "id": "human",
                "name": "Human play 1v1",
                "description": "The same classic rules paced at two turns per second.",
                "game_config": {
                    "players": players, "max_turns": 1200,
                    "tick_interval_seconds": 0.5, "turn_timeout_seconds": 1,
                },
            },
        ],
        "certification": {
            "game_config": {"players": players, "seed": 7, "max_turns": 40, "tick_interval_seconds": 0.05},
            "players": [{"player_id": "expander"}, {"player_id": "expander"}],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = ROOT / "coworld_manifest_template.json"
    content = json.dumps(template(), indent=2) + "\n"
    if args.check:
        if not output.exists() or output.read_text() != content:
            raise SystemExit("manifest is stale: run python -m integrations.softmax.tools.manifest")
        print("Coworld manifest matches runtime schema and docs.")
    else:
        output.write_text(content)
        print(output)


if __name__ == "__main__":
    main()
