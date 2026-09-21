"""Run a complete local Coworld episode without Docker or a Softmax account."""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-turns", type=int, default=1200)
    parser.add_argument("--variant", choices=("competition", "ffa", "castles"), default="competition")
    parser.add_argument("--human", action="store_true", help="control slot 0 in the browser")
    parser.add_argument("--keep-open", action="store_true", help="keep the server open for replay inspection")
    parser.add_argument("--output", type=Path, default=ROOT / "integrations/softmax/local-output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Each run gets its own directory so old success artifacts cannot mask failure.
    run_dir = output / f"episode-{time.time_ns()}"
    run_dir.mkdir(mode=0o700)
    count = 4 if args.variant == "ffa" else 2
    tokens = [secrets.token_urlsafe(24) for _ in range(count)]
    names = ["Red", "Blue", "Green", "Purple"][:count]
    config = {
        "tokens": tokens,
        "players": [{"name": "You" if args.human and slot == 0 else name} for slot, name in enumerate(names)],
        "ruleset": "build_castles" if args.variant == "castles" else "classic",
        "seed": args.seed,
        "max_turns": args.max_turns,
        "tick_interval_seconds": 0.5 if args.human else 0.0,
        "turn_timeout_seconds": 1 if args.human else 0.5,
    }
    config_file = run_dir / "config.json"
    config_file.write_text(json.dumps(config))
    config_file.chmod(0o600)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "COGAME_CONFIG_URI": config_file.as_uri(),
        "COGAME_HOST": "127.0.0.1",
        "COGAME_PORT": str(args.port),
        "GENERALS_KEEP_OPEN": "1",
        "COGAME_RESULTS_URI": (run_dir / "results.json").as_uri(),
        "COGAME_SAVE_REPLAY_URI": (run_dir / "replay.json").as_uri(),
        "COGAME_PLAYER_FAILURE_URI": (run_dir / "failure.json").as_uri(),
    }
    processes = []
    base = f"http://127.0.0.1:{args.port}"
    try:
        server = subprocess.Popen([sys.executable, "-m", "integrations.softmax.server"], cwd=ROOT, env=env)
        processes.append(server)
        with httpx.Client(timeout=1) as client:
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError("game server exited during startup")
                try:
                    if client.get(base + "/healthz").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("game server did not become ready within 60 seconds")
        for slot in range(count):
            query = urlencode({"slot": slot, "token": tokens[slot]})
            if slot == 0 and args.human:
                print(f"Play: {base}/client/player?{query}", flush=True)
            else:
                player_env = {**os.environ, "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{args.port}/player?{query}"}
                module = "builder_player" if args.variant == "castles" else "player"
                processes.append(subprocess.Popen(
                    [sys.executable, "-m", f"integrations.softmax.{module}"], env=player_env, cwd=ROOT))
        print(f"Watch: {base}/client/global", flush=True)
        print(f"Artifacts: {run_dir}", flush=True)
        deadline = time.monotonic() + 900
        while not (run_dir / "results.json").exists():
            if server.poll() is not None or (run_dir / "failure.json").exists():
                raise RuntimeError("episode failed; inspect the server output and failure artifact")
            if time.monotonic() > deadline:
                raise RuntimeError("episode exceeded the local deadline")
            time.sleep(0.1)
        print((run_dir / "results.json").read_text(), flush=True)
        for player in processes[1:]:
            if player.wait(timeout=5) != 0:
                raise RuntimeError("a bundled player failed")
        if args.keep_open or args.human:
            print(f"Replay: {base}/client/replay (Ctrl+C to stop)", flush=True)
            while server.poll() is None:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
