# Generals Competition · Softmax Coworld

A bounded **1v1** territory-control game, running this repository's
`GeneralsEnv(mode="competition")` rules. The authoritative engine is Python/JAX
on CPU. Players run separately and connect over WebSocket. A lightweight bridge
lets existing Python, C++, and Rust competition bots keep their stdio protocol.

## Play locally

From the repository root, with Python 3.12:

```bash
pip install -e '.[softmax]'
python -m integrations.softmax.local --human
```

Open the printed player link. Select an owned tile, then use arrow keys/WASD or
click neighboring cells to queue a route. Outlined yellow arrows show queued moves;
a white arrow marks the move already submitted for this turn. E undoes the last
queued action; Q clears all remaining actions. Neither cancels an action already
submitted. H toggles half-army moves for new inputs, B queues a castle build,
and Space queues a pass. Moves execute one per turn. If a move cannot execute
(including insufficient army), fails to secure its destination, or an obstacle
blocks the queued route, the entire queue is cancelled and the selection is
cleared. Click an owned tile to start again; the general is never selected
automatically. Mountains and fog obstacles cannot be queued into. A castle
hidden by a fog obstacle can be entered once revealed.
The browser passes automatically if you do not act. Queues are local
to the browser and reset on disconnect or reload. The human
configuration advances at two turns per second. After the match, use the same
page to watch the full replay. Ctrl+C stops the local server.

For two bundled bots playing as fast as they can:

```bash
python -m integrations.softmax.local --seed 7 --keep-open
```

`--max-turns 40` makes a short smoke test; the competition variant always uses
1,200. Each local run writes into its own ignored `local-output/episode-*`
directory. `config.json` contains local player tokens; do not publish it. The
shareable outputs are `results.json` and `replay.json`. `--port` selects another
local port. The local launcher binds only to loopback.

## Rules and scoring

- Two opposing generals on an independently sampled 18–21 by 18–21 board.
- Fog of war: a player sees its owned tiles and their neighbors. Public army
  and land totals remain visible. Unexplored mountains/castles share one
  structure marker. Enemy generals are hidden until visible.
- A move sends all but one army, or half the source army (rounded down), into
  one orthogonally adjacent cell. Mountains cannot be entered. Friendly armies
  combine; attacking armies subtract from defenders. You must exceed the
  defending army to take a cell under normal combat.
- Generals and owned castles grow each even tick; owned land grows every 50
  ticks. There are no neutral castles at spawn.
- Build on an owned plain cell for **35 + Σ max(0, 14 − 2d)** armies, where `d`
  is Manhattan distance to each of your existing castles/general. Builds resolve
  before moves. Unaffordable/illegal game moves are no-ops.
- From turn 800, Deathtouch makes an executed move onto the enemy general an
  immediate victory. All move-order and simultaneous-capture rules come directly
  from the shared engine; the adapter does not reimplement them.
- Capture scores **+1** for the winner and **−1** for the loser. Reaching the
  1,200-turn cap scores **0 / 0**, regardless of army or land advantage.

The hosted runtime adds explicit failure rules: actions have a 500 ms deadline
(including transport) from publication of each observation. A missing action is
a pass. After 20 consecutive missed turns a player forfeits; if both reach the
threshold together, both score zero. A valid pass resets the counter. If a
player never connects before the 180-second start deadline, the episode emits a
typed player failure instead of competitive scores. Reconnection is allowed
during play with the original token, but never resets timeout counters.

Competitive episodes omit `seed`, generating a fresh server-private 32-bit seed.
Explicit seeds are for reproducible local tests/evaluations. The seed is recorded
only in the completed replay. Do not pin a league to a known fixed seed: a bot
could reconstruct the hidden map.

## Player protocol

See [PLAYER_PROTOCOL.md](PLAYER_PROTOCOL.md) for the exact JSON contract and
the existing [competition protocol](../../competition/protocol.py) for stdio bots.
The bundled player is the repository's pure-Python Expander; it does not need
JAX or access to the engine.

To bridge another bot, run in its player container:

```bash
python -m integrations.softmax.player -- /path/to/compiled-bot
```

The runner supplies `COWORLD_PLAYER_WS_URL`. Python scripts can be passed as
`-- python -u /path/to/main.py`. Build C++/Rust executables into the player image
ahead of time. The bridge owns only its child process; the game owns scoring.

## Spectators and replays

`/client/global` shows live public scores and match status. **It deliberately
does not show the live board.** Player containers can reach the game server, so
an unauthenticated omniscient WebSocket would defeat fog of war. Player pages
and `/player` require the slot's token. The public stream accepts no game-control
commands, and `/replay.json` is unavailable until all success artifacts are saved.

The completed replay includes every board frame and the submitted/applied
actions, seed, timeout flags, and result. The tile renderer and styles from `generals-competition` serve player views
and replays. Crown, castle, and mountain sprites come directly from this repo's
`generals/assets/images`, with the same assets copied into the static bundle. The static replay bundle needs no Python server, JAX, WASM,
or network access beyond fetching its replay and local assets. It supports
autoplay, looping, pause, seek, playback speed, resize, and gzip replay bytes.
Historical releases keep their own immutable viewer bundle.

For a standalone replay, serve the generated bundle and a replay over HTTP:

```bash
integrations/softmax/tools/build_replay_viewer.sh integrations/softmax/dist/replay-viewer
# Copy only a completed replay into dist/replay-viewer/replay.json, then:
python -m http.server 8090 --directory integrations/softmax/dist/replay-viewer
# Open http://localhost:8090/#replay=replay.json
```

See [GLOBAL_PROTOCOL.md](GLOBAL_PROTOCOL.md). Full live board spectating would
require a separate trusted-viewer authorization contract with Softmax; it must
not be enabled by copying the example's public omniscient stream.

## Build and certify

The project lives in this directory so the training package does not depend on
the Coworld SDK. It needs Docker with Compose/Buildx and Docker daemon access.
Keep the SDK in a separate Python environment. The initial integration targets
the public Coworld contract at commit
`4c26e51` (2026-09-08); pin and review SDK updates before releases.

From the repository root:

```bash
python -m integrations.softmax.tools.manifest --check
coworld build --project integrations/softmax --version 0.1.0
coworld certify integrations/softmax/dist/coworld_manifest.json
```

`compose.yaml` builds separate CPU game and lightweight player images. The
manifest template embeds the protocol/rules docs and derives its config schema
from the runtime model. If those sources change, regenerate it with
`python -m integrations.softmax.tools.manifest`. The build hook recreates the
static replay directory and Coworld resolves images to immutable identifiers.

Before publishing, commit and push the release inputs on the integration branch
so the pinned source URL resolves, inspect a full-length match and static replay,
and confirm the game name/account ownership with Softmax. Then use the
authenticated SDK:

```bash
coworld upload-coworld integrations/softmax/dist/coworld_manifest.json
```

Hosted verification must exercise real episodes, result scores, player shutdown,
logs, and the hosted replay. A local pass is not evidence of hosted operation.
League setup uses Softmax's platform scheduler; no custom commissioner,
reporter, grader, or optimizer is required for this release.

## Maintenance and tests

```bash
JAX_PLATFORMS=cpu pytest -q tests/test_softmax.py tests/test_matchup.py
```

The integration tests cover real engine observation/wire parity, action
validation, authentication, information boundaries, deterministic replay,
timeouts, startup failures, Deathtouch scoring, and failed artifact writes.
Before releases, run the complete engine suite, local container certification,
and browser checks for live player controls and the static replay bundle.

To run the real Chromium checks (including mobile layout, gzip, corrupt replays,
and WebSocket pong), install `playwright` and Chromium, then run:

```bash
GENERALS_BROWSER_TESTS=1 pytest -q tests/test_softmax_browser.py
```

Set `CHROMIUM_PATH` if Chromium is not on PATH, or use Playwright's installed
browser. Deployment dependencies, including transitive packages, are pinned in
`requirements.lock`; regenerate it with `uv pip compile requirements.txt
--python-version 3.12 --output-file requirements.lock` from this directory when
deliberately updating the runtime.

Rules changes belong in `generals/`; both this adapter and the local competition
runner call `generals.core.match`. Publish a new Coworld version for rules,
protocol, rendering, or dependency changes. Preserve old replay fixtures when
the format changes. Keep participant support and league balancing distinct from
technical adapter maintenance. Hosting/resource allowances and ongoing support
ownership need agreement with Softmax.

## Visual sources

`static/board.js` reuses the tile-rendering function from
`generals-competition/board.js` at commit `83a9b23`. `static/board.css` contains
that repo's tile styles and color variables from the same commit. Only the
Coworld snapshot-to-tile mapping and selected-cell indicator are adapter-specific.
The surrounding layout follows its `assets/site.css` replay viewer. There is no
procedural game fallback or second game simulation in this renderer.

Keep the tile styling aligned with the competition site when updating it.
Sprites stay owned by `generals/assets/images`, and the Quicksand font comes
from `generals/assets/fonts`, including its license in the replay bundle. No
copied source assets need to be synchronized manually.
