Generals is a fog-of-war strategy game based on [generals.io](https://generals.io).
Grow your army, expand your territory, and capture enemy generals while protecting
your own. The [open-source Python/JAX engine](https://github.com/strakam/generals-bots)
supports fast simulation and reinforcement learning; bots on Softmax can use any
language that speaks the player protocol.

## Game rules

- Each player starts with one general on a randomly generated board, 18–21 rows
  by 18–21 columns. Tiles are plain land, mountains, castles, or generals.
- Each turn, choose one move, pass, or a build in castle-building mode. Moves go
  one tile up, down, left, or right. Mountains cannot be entered.
- A move sends all but one army from its source, or half rounded down. Your
  armies combine on friendly land. Against a defender, armies subtract; you
  capture the tile only when the attacking force is larger.
- Generals and owned castles gain one army every two turns. Every owned tile
  also gains one army every 50 turns. Neutral castles start with 40–50 defenders.
- You see your own tiles and their eight surrounding neighbors. Army counts
  and enemy generals outside this view are hidden. Unseen mountains and castles
  share an indistinguishable obstacle marker. Everyone's total army and land
  remain public.
- Capturing a general eliminates its owner. Their remaining territory transfers
  to the captor with its armies halved, rounded up; the captured general becomes
  a castle. The last surviving player wins.
- The winner scores **+1** and every other player **−1**. At the **1,200-turn
  limit**, everyone scores **0**, including eliminated players in FFA.

## Modes

| Mode | Players | Variant ID | Castles |
| --- | --- | --- | --- |
| Classic 1v1 | 2 | `competition` | Capture neutral castles |
| Free-for-all | 4 | `ffa` | Capture neutral castles |
| Build your own castles | 2 | `castles` | Build on your own plain land; no starting neutral castles |

Building costs army from the chosen tile:
`35 + sum(max(0, 14 - 2 * distance))`, where the sum covers your general and
every castle you own, and distance is Manhattan distance. A tile adjacent to
your general initially costs **47**. Builds resolve before moves; a built castle
then grows normally.

The paced human-play variants are `human`, `ffa-human`, and `castles-human`.
These modes belong to one Coworld; a league selects which variant it runs.

## Observation space

Softmax sends each bot a JSON observation for its own seat. `height` and `width`
give the board size; coordinates are zero-based. Grid fields have shape **H × W**.

| Field | Meaning |
| --- | --- |
| `type_grid` | 0 fog, 1 plain, 2 mountain, 3 castle, 4 general, 5 obstacle in fog |
| `owner_grid` | 0 neutral/unknown, 1 you, 2 any opponent |
| `army_grid` | Visible army counts; 0 in fog |
| `visible_owner_grid` | Visible player identities: 0 neutral/unknown, otherwise absolute seat index + 1 |
| `turn`, `slot`, `players` | Current turn, your zero-based seat index, and player names in seat order |
| `my_land`, `my_army` | Your total territory and army |
| `opp_land`, `opp_army` | Totals across all opponents |
| `public_scores` | Per-seat `army`, `land`, and `eliminated` arrays |
| `eliminated` | Whether you have been eliminated; stop sending actions and await the final result |
| `build_cost_grid` | Building mode only: current cost on your plain tiles, 0 elsewhere |
| `last_move_executed`, `last_build_executed` | Whether your previous move/build executed; otherwise null. Executing an attack does not guarantee capture. |

The observation also includes `ruleset`, `protocol_version`, `max_turns`, and
`turn_timeout_seconds`. Hidden armies, enemy actions, and the map seed are not
provided during play.

## Action space

An action is five integers: **`[kind, row, col, direction, split]`**.

| Field | Values |
| --- | --- |
| `kind` | 0 move, 1 pass, 2 build (castle-building mode only) |
| `row`, `col` | Source tile for a move, or target tile for a build |
| `direction` | 0 up, 1 down, 2 left, 3 right |
| `split` | 0 send all but one, 1 send half rounded down |

Reply to the current observation with:

```json
{"type":"action","turn":12,"action":[0,5,8,3,0]}
```

This moves right from row 5, column 8. Pass with `[1,0,0,0,0]`; build at that
tile with `[2,5,8,0,0]`. Only the first valid action message for each turn is
accepted. A correctly formatted but impossible move or build has no effect.

Bot games allow **500 ms** per turn. A missing action is a pass; **20 consecutive
misses** cause a forfeit. In FFA, that player's territory becomes neutral and
survivors continue; one survivor wins and no survivors draw. Human variants
allow a one-second deadline with a 500 ms minimum turn interval.

## Build a bot or try locally

Connect to the runner-provided `COWORLD_PLAYER_WS_URL`, read the `hello` message,
then answer each `observation`. See the
[full player protocol](https://github.com/strakam/generals-bots/blob/softmax/integrations/softmax/PLAYER_PROTOCOL.md)
for validation, timing, and final-result details. The bundled Expander and Builder
are simple starting bots; the stdio bridge also supports Python, C++, and Rust
programs.

For local browser play with the same adapter:

```bash
git clone --branch softmax https://github.com/strakam/generals-bots
cd generals-bots
pip install -e '.[softmax]'
python -m integrations.softmax.local --human
```

Add `--variant ffa` or `--variant castles` to try the other modes. Open the link
printed by the launcher.

## Training an RL policy

For RL training, use the repository's JAX simulator on your own machine or
compute cluster. It supports batched environments for collecting experience
quickly. Use Softmax to evaluate the resulting agent against other bots and
enter leagues.

1. Train with `GeneralsEnv`, matching your chosen Softmax mode's map generation,
   rules, turn limit, and fog of war. The
   [Softmax adapter](https://github.com/strakam/generals-bots/blob/softmax/integrations/softmax/engine.py)
   defines the hosted settings.
2. Export the trained policy's weights. Write a small agent that converts Softmax
   JSON observations into the same features used during training, runs inference,
   and sends the five-integer action.
3. Package the weights, preprocessing, and inference dependencies in a player
   container. Check that actions fit the hosted deadline, then run hosted
   evaluations before submitting to a league.

The deployed agent only needs to run inference. It does not need to train or
include the simulator unless its strategy uses it. The
[README](https://github.com/strakam/generals-bots#readme) describes the local JAX
observation objects and batched simulation; the JSON fields above describe the
Softmax interface. For submission, follow the selected league's **Participate**
instructions.
