# Replay agreement: does the engine reproduce real generals.io games?

`replay_agreement.py` feeds every game in the public replay archive
(`strakammm/generals_io_replays`, 18,803 1v1 replays, HuggingFace) through the
JAX engine move for move and checks two things a wrong engine cannot fake:

* **legality** — is each recorded move still playable on the simulated board at
  the moment it resolves? A human only ever submits moves that are legal on the
  board they see, so the first illegal recorded move marks the first tick on
  which the simulated board no longer matches the one the players saw.
* **the ending** — every replay in the archive ends with a general capture.
  Does the engine declare the same winner on the same tick?

Results in `results/`: `per_game.csv` (one row per replay), `summary.json`,
`excluded.csv` (empty: nothing was excluded).

## Protocol

**Data.** All 18,803 replays are 2-player (`len(usernames) == 2`), have two
on-board generals, non-overlapping generals/cities/mountains, city garrisons of
40–50 (encodable: the engine reads grid values > 2 as castles), moves sorted by
turn, no duplicate (player, turn) pairs, no non-adjacent moves, and a final move
that lands on the opponent's general. The dataset carries two replay-format
versions: 5 (10,613 games, older) and 13 (8,190 games). The parquet has no
`afks`, `teams`, `swamps`, `deserts` or modifier columns, so no such feature
could be detected; none of the checks below needed one.

**Grid.** `tile = row * mapWidth + col`. Engine grid encoding
(`generals.core.game.create_initial_state`): `0` empty, `-2` mountain, `1`/`2`
P0/P1 general, `>2` castle with that army. Every grid is padded to 23×23
(largest map side) with mountains so the JIT compiles once per tick bucket;
mountains are inert.

**Moves.** `[playerIdx, startTile, endTile, is50, turn]` →
`[0, row, col, direction, is50]` with direction `0=UP 1=DOWN 2=LEFT 3=RIGHT`.
Ticks without a move for a player get `[1, 0, 0, 0, 0]` (pass).

**Alignment (verified before scaling up).** A move recorded with `turn == t` is
played in the engine step taken from `state.time == t`, i.e. the (t+1)-th call
of `game.step`. Both counters start at 0; the first production lands on tick 2
in both (`global_update` increments structures on even ticks; see the comment in
`game.py`). Checked on the first 5 replays with plain `game.step`
(`--selftest 5`): the general holds exactly `1 + t // 2` armies when the first
move is played (t = 16, 22, 9, 13, 24 → 9, 12, 5, 7, 13), army totals are
plausible throughout, and all five games end by the recorded capture on the
recorded tick with the recorded winner. Consistent with this, the earliest first
move in the whole archive is at turn 2 (never 0 or 1), which is the first tick a
general has 2 armies. The `lax.scan` used for the full run was also checked to
give the same endings as plain `game.step` on those games.

**Policies.** A move by a player who is already eliminated when it resolves
(general fell earlier in the same tick) is not counted as illegal. Moves
recorded after the simulated game has ended are counted as
`*_n_moves_after_end`, not as illegal. Duplicate (player, tick) pairs would keep
the first move and count the rest (`n_dropped_duplicate`); there are none.
Legality is judged in resolution order (each move on the board as the previous
one left it) with `generals.core.action.compute_valid_move_mask`; the pre-tick
definition (both moves judged on the board before the tick) is also reported
(`*_n_illegal_prestep`).

**Two move-resolution rules, simulated side by side from the same actions.**

| rule | what it is | where |
|---|---|---|
| **current** (default) | chasing > reinforcing > *smaller* army first, ties by player index | `generals/core/game.py::_determine_move_order` (current master) |
| **legacy** (`legacy_move_priority=True`) | priority alternates every tick: P0 resolves first on even ticks, P1 on odd ticks, regardless of the moves | same function, flag added on this branch; it is the rule of `generals-bots` before commit `e5676c3` (2025-04-10, "Add new move priority system based on official generalsio"): `agent_order = agents[:]` reversed after every `step` |

The archive predates the change in generals.io's rule. The flag is opt-in
(`game.step(state, actions, legacy_move_priority=True)` or
`GeneralsEnv(legacy_move_priority=True)`); `tests/test_legacy_move_priority.py`
checks that the default is unchanged.

Because the legacy run reproduces the archive essentially exactly (below), the
first tick on which the two simulated boards differ is the exact first
divergence of the current rule from the recorded game — no per-tick recorded
state is needed. Ticks on which both games are already over are ignored (the
spoils of the final capture are halved in whichever order the last tick
resolved).

## Results (engine commit 6055d5a; runtime 125 s on 10 CPU workers)

18,803 games used, 0 excluded, 9,275,253 recorded moves (39,716 splits),
6,161,546 ticks simulated; both players moved on 3,755,216 ticks and their
moves touched a common cell on 49,880 ticks (0.8 % of ticks, 2.65 per game).

| | current rule | legacy rule |
|---|---|---|
| games reproduced exactly (0 illegal moves, same winner on the same tick) | **12,244** (65.1 %) | **18,590** (98.9 %) |
| games with ≥1 illegal move | 6,471 | 186 |
| illegal moves (of 9,275,253) | 51,717 (0.56 %) | 674 (0.0073 %) |
| … source not owned / <2 armies / into a mountain | 45,540 / 6,176 / 1 | 221 / 452 / 1 |
| ended on the recorded tick (±1 gives the same count) | 17,625 (93.7 %) | 18,756 (99.75 %) |
| … of which same winner | 17,624 | 18,756 |
| ended early | 79 (24 with a different winner) | 13 (4 different winner) |
| ended late | 0 | 0 |
| never ended | 1,099 | 34 |

By format version: v13 (8,190 games, 4,537,351 moves) — legacy rule
**8,190 / 8,190 exact, 0 illegal moves**; current rule 4,566 exact, 33,026
illegal. v5 (10,613 games, 4,737,902 moves) — legacy 10,400 exact, 674
illegal moves in 186 games; current 7,678 exact, 18,691 illegal.

### Why the current rule disagrees (6,559 games)

6,424 of the 6,559 have a tick on which the current-rule board departs from the
legacy board while at least one of the two games is still on; the other 135 are
the version-5 residual below (the two boards never differ, the disagreement is
inherited from the legacy run). On every one of the 6,424 the two players'
moves touched a common cell (6,248 one moved into the cell the other was
leaving, 172 both moved into the same cell, 4 head-on) and the two rules put a
different player first (`divergence_where_legacy_and_current_first_mover_coincide = 0`).
The first illegal move follows the divergence on the same tick in 4,345 games,
within 1 tick in 5,244, within 2 in 5,454, within 10 in 5,843 (of 6,332 with an
illegal move after a divergence).

First-divergence mechanism (from `summary.json`, `first_divergence_detail`):

| games | what happened on the divergence tick |
|---|---|
| 3,998 | **chase, chaser overruns**: A moves onto the cell B is leaving; current lets the chaser go first and it beats B's full stack, so B's own move fails (its army is gone). Legacy let B leave first; A took the 1 army left behind. |
| 2,228 | **chase, chaser bounces**: same geometry, but the chaser's army is not bigger than B's full stack, so the attack fails and B walks away with a reduced stack; legacy: A took the cell. |
| 86 | both moved into the same neutral cell with equal armies (tie now goes to P0 by index; legacy to whoever had priority that tick) |
| 84 | both moved into the same cell, one reinforcing its own cell, the other attacking it (reinforce now resolves first) |
| 22 | chase onto a departing **general** (chaser bounces off; in the recorded game it took the cell the general's stack had just left) |
| 4 | head-on swap (each moved into the other's source; smaller army now first) |
| 2 | both moved into the same neutral castle with unequal armies (bigger now resolves last) |
| 135 | no divergence from the legacy board (v5 residual, see below) |

Consequence for the game: 5,380 games still end on the recorded tick with the
recorded winner (the desynchronised cell was tactically irrelevant); 1,099 never
end (the recorded final capture no longer succeeds); 55 end early with the same
winner and 24 early with the other winner; 1 (`yLw-cvna2`) ends on the same tick
with the other winner.

### Residual under the legacy rule (213 games, all format version 5)

Under the alternating rule every version-13 game is reproduced exactly. The
213 version-5 misses split as: 166 that still end on the recorded tick with the
recorded winner but contain recorded moves that are illegal on the simulated
board, and 47 that end differently (34 never, 13 early). The illegal moves look
like client-queued moves the server rejected — 452 are from a cell the mover
owns with 1 army (typically the tile the player had emptied on the previous
tick), 221 from a cell the mover no longer owns, 1 into a mountain
(`SKtCdGo8e`, tick 86) — which suggests the version-5 logger recorded submitted
rather than executed moves; the newer format contains no such move. 61 of the
"not owned" moves (42 games) start from a cell the mover never held on the
simulated board, so some version-5 logs are also missing a move or resolved an
earlier clash differently; 36 of the 213 have a same-destination, head-on or
chase tick within 2 ticks before the first illegal move, hinting at a different
tie rule in the 2016–17 server for those cases. This residual (1.1 % of games,
2.0 % of v5 games) is not resolved.

### Heuristic without the legacy oracle

For the paper's question "is the first divergence a move-order-sensitive
tick?", the exact answer via the legacy oracle is 6,424 of 6,424. The
approximate answer using only the recorded moves — an interaction (the two
moves share a cell) within k ticks *before or on* the first illegal move — is:
on the tick itself 5,030 games, within 1 tick 5,591, within 2 ticks 5,791,
within 5 ticks 6,063, within 10 ticks 6,166 (of 6,471 games with an illegal
move under the current rule). The gap is games where the desynchronised cell
only mattered many ticks later.

## Reproducing

```
uv venv --python 3.12 ~/.cache/generals-validate-venv
VIRTUAL_ENV=~/.cache/generals-validate-venv uv pip install -e . pyarrow numpy
JAX_PLATFORMS=cpu ~/.cache/generals-validate-venv/bin/python paper/validation/replay_agreement.py --selftest 5
JAX_PLATFORMS=cpu ~/.cache/generals-validate-venv/bin/python paper/validation/replay_agreement.py --workers 10
```

The full run takes about 2 minutes on 10 CPU workers (one `lax.scan` per
game, 11 JIT compilations per worker, persistent compilation cache in
`~/.cache/generals-validate-jax-cache`). Disagreeing games can be inspected at
`https://generals.io/replays/<id>`; ids are in `per_game.csv`
(`cur_agrees == False`) and sampled by outcome in `summary.json`.
