# my_bot — iteration log

Each generation is a separate agent directory under `competition/agents/`, so
every earlier bot stays runnable and every claim below stays reproducible.
`my_bot` is generation 1; nothing is edited in place.

Measurement: `competition/arena.py` plays every seed in **both seat orders** (so
seat advantage cancels) under `--mode competition`.

```bash
python competition/arena.py my_bot6 expander_python --seeds 6
python competition/arena.py my_bot6 my_bot5 --seeds 6      # vs the previous gen
```

Compare against a **fixed** opponent, not against the previous generation only:
two similar bots that both refuse to attack draw every game, which hides real
differences. `expander_python` is the stable reference below.

---

## Scoreboard — 12 matches vs `expander_python` (6 seeds × 2 seats)

| Gen | Idea added | W–L–D | Median turns | Avg land | Avg army | Avg castles |
|---|---|---|---|---|---|---|
| `my_bot` | castle building (capped at 3) | 1–1–10 | 1200 | 37.3 | 2,337 | 2.8 |
| `my_bot2` | uncap castles | 0–1–11 | 1200 | 39.8 | 3,521 | 5.4 |
| `my_bot3` | every structure saves, not just the general | 0–1–11 | 1200 | 77.0 | 10,607 | 46.2 |
| `my_bot4` | BFS attack + deathtouch | 10–0–2 | 786 | 190.2 | 5,643 | 52.5 |
| `my_bot5` | prune stale enemy memory + defend general | 10–0–2 | 786 | 190.2 | 5,643 | 52.5 |
| `my_bot6` | scout *behind* enemy territory | **12–0–0** | **570** | 167.6 | 2,471 | 35.0 |
| `my_bot7` | stop economy at t500, mass for the strike | (see below) | — | — | — | — |
| `my_bot8` | deathtouch race: rush the *nearest* unit | (see below) | — | — | — | — |
| `my_bot9` | route through fogged structures | **38–2–0** | — | — | — | — |

`expander_python` finishes the `my_bot6` matches on 0 land, 0 army — swept off
the board in all 12.

### Generations 7–9, measured against a fixed opponent

Once bots get close in strength, head-to-head runs stop separating them (two
similar bots mostly draw) and 12-match samples are pure noise. Generations 7–9
were therefore scored against a frozen `my_bot5`, over **120 matches** (60 seeds
× 2 seats), and every result was replicated on a disjoint seed range.

| Gen | seeds 0–59 | seeds 100–159 | draws |
|---|---|---|---|
| `my_bot6` | 65W–26L–29D (0.662) | 65W–26L–29D (0.662) | 29 |
| `my_bot7` | — | 68W–25L–27D (0.679) | 27 |
| `my_bot8` | — | 63W–27L–30D (0.650) | 30 |
| `my_bot9` | **81W–27L–12D (0.725)** | **79W–27L–14D (0.717)** | **12–14** |

Only `my_bot9` moves. Draws are cut by more than half, and it is the first
generation to beat `my_bot6` head-to-head (61W–54L–5D, though that margin alone
is inside the noise).

Head-to-head against the previous generation (12 matches each):

| Matchup | Result |
|---|---|
| `my_bot6` vs `my_bot5` | **8W–2L–2D** |
| `my_bot5` vs `my_bot3` | 4W–0L–8D |
| `my_bot4` vs `my_bot3` | 4W–0L–8D |
| `my_bot3` vs `my_bot2` | 0W–1L–7D (but 46 vs 5 castles) |

---

## The two facts that drive everything

Both read straight out of the engine, and both are counter-intuitive:

1. **A castle is worth ~25 land.** `global_update` in `generals/core/game.py`
   grants +1 army per owned cell every **50** ticks, but +1 per general/castle
   every **2** ticks. So land is 0.02 army/turn/cell and a castle is
   0.5 army/turn. A ~40-army castle repays itself in ~80 turns and compounds
   for the rest of a 1200-turn match. Competition maps spawn with *no* neutral
   castles, so building is the only economy that exists.
2. **From turn 800, touching the enemy general wins.** Deathtouch
   (`generals/modifiers/deathtouch.py`) makes any move that executes onto the
   enemy general a win *regardless of army*. After turn 800 the game is not
   about material at all — it is about knowing where their general is and
   having any unit able to walk there.

---

## Generation notes

### `my_bot` — castle building
The starter expander never emits action kind `2`, so it builds nothing and the
competition board stays castle-free all game. Naively adding "build when a cell
can afford it" fires **zero** times: the expander spends every stack the moment
it is the best move, so its biggest owned-plain stack peaks around 24–30 army
and never reaches the 35 minimum. Building requires deliberate saving:
`_stage_general` withholds the general from the expander scan, then walks its
savings one cell out (a general can't be built on) so `_find_build` can cash it
in next turn. Result: 2–3× the army of the baseline — and still no wins.

### `my_bot2` — uncap castles
`MAX_CASTLES` 3 → 99, `LAST_BUILD_TURN` 700 → 1050, justified by the 25:1
income ratio. Army rises 2,337 → 3,521, but castles only reach 5.4: the cap was
never the binding constraint.

### `my_bot3` — compounding economy
The real constraint was that only the *general* saved. Castles earn at the same
+1-per-even-tick rate, so `_stage_from_structures` lets every structure fund the
next one. Castles 5.4 → **46.2**, army 3,521 → **10,607**. Still 0 wins: a 10k
army that never walks anywhere is just a number at turn 1200.

### `my_bot4` — the attack
Adds `_first_step` (BFS over passable cells, ~484 cells, cheap enough per turn),
persistent memory of the enemy general once seen, and a turn-priority change:
builds and staging are rare events, so **every other turn goes to marching the
biggest stack at the enemy** instead of aimless expansion. Before turn 800 it
requires army > general's army + margin; after 800 any stack will do, because
contact alone wins. 0 wins → **10 wins**.

### `my_bot5` — memory hygiene and defense
Two bugs/gaps found by instrumenting a drawn seed:
- `seen_enemy` was never pruned, so the attack marched at cells it *already
  owned*, arrived, and bounced back to the previous one forever.
  Now cells visible and no longer enemy-held are discarded.
- `_defend`: past turn 800 an enemy parked next to our general is lethal.
  Moving onto their cell is a "chase", which the engine resolves *first* —
  capture the source and the touch never executes.

**A measured failure worth keeping:** committing to one marching stack across
turns (rather than re-picking the biggest each turn) *sounded* right — it fixes
visible oscillation — but cost 3 of 4 wins against `my_bot3` (4W → 1W). A
depleted stack keeps its commitment and walks around achieving nothing, while
"always the biggest" self-corrects every turn. Reverted; the comment in
`_spearhead` records why so it doesn't get re-introduced.

### `my_bot6` — scout behind enemy lines
The remaining draws were all the same failure: `enemy_general=None` for the
entire game. Targeting the *nearest* enemy cell just nibbles the shared border —
take a frontier cell, the next nearest enemy cell is one step away, and the
attack never penetrates far enough to reveal anything. `_scout_targets` instead
aims at **fog cells adjacent to known enemy territory**, pushing the search
inward toward where their general must be; with no contact yet it sweeps the far
half of the board (generals are ≥14 cells apart). 4W–0L–8D → **11W–0L–1D**
against `my_bot3`, and a clean 12–0–0 against the baseline.

### `my_bot7` — mass for the strike (no measurable gain)
This was the roadmap's **top-ranked** idea: stop building at turn 500 once the
enemy general is known, and spend the freed turns walking the second-biggest
stack into the spearhead (`_gather`, rallying on a *cell* so the target cannot
run away — the mistake `my_bot5` made). A 30-seed run looked like a clear win
(0.700 vs 0.642); at 60 seeds it collapsed to 0.679 vs 0.662, inside the
interval. Ablation showed the gather itself contributes essentially nothing:
build-cutoff-only scored 0.692 and gather-only scored 0.642, exactly matching
`my_bot6`. Gathering has nothing to move, because with building enabled every
structure converts its savings into a castle before a stack can accumulate.
Kept (harmless, and it is the mechanism that uses the freed army), but **not
demonstrated**. The lesson is the harness one: 30 seeds is not enough.

### `my_bot8` — deathtouch race (no gain)
After turn 800 contact wins regardless of army, so send whichever unit is
*fewest steps* from the enemy general rather than the biggest one — one BFS out
from their general ranks every cell at once. Sound reasoning, no effect: 0.650,
indistinguishable from `my_bot6`. Instrumenting a drawn game showed why, and it
had nothing to do with unit choice.

### `my_bot9` — fogged structures are not bedrock
The diagnostic that mattered. In a drawn game the log read:

```
turn=800  gen_known=True  nearest_unit_dist=None
turn=1100 gen_known=True  nearest_unit_dist=None
```

The general's position was known for 500 turns and **no path to it existed**.
Checking the board explained it — all four neighbours of the enemy general were
type 5:

```
enemy general (10,18); neighbours: [((9,18),5), ((11,18),5), ((10,17),5), ((10,19),5)]
```

Type 5 is "structure under fog", which is a mountain **or a castle** — and a
castle is capturable. Every generation since `my_bot4` inherited the starter's
`_is_passable`, which lumps 5 in with mountains, so enemy castles were treated
as bedrock. Bots ring their own general with castles (adjacent builds cost only
43), so each side had accidentally made itself unreachable. The attack was never
choosing badly; it had no legal plan at all.

`_routable` now plans strictly first and retries allowing fog-structures if
nothing is reachable. Guessing wrong costs one turn (the engine turns a move
into a mountain into a pass), so `_note_attempt` records bounces and blacklists
a cell after two — self-correcting, needing no map knowledge.

Draws 29 → 12. This is the largest single gain since `my_bot6`, and it was a
**bug**, not a strategy.

---

## Not yet implemented — ranked by expected value

Items 1 and 2 were attempted in `my_bot7`/`my_bot8` and did not pay — the list
below is what is left, re-ranked by what the `my_bot9` diagnosis suggests.

1. **Don't wall in your own general.** `my_bot9` fixed our *attack* against
   enclosed generals, but we still enclose our own — `_stage_general` builds
   adjacent (cost 43) which is exactly what created the wall. Building 5+ cells
   out is cheaper (35) *and* keeps our general approachable, which matters
   because the same fix is available to any opponent.
2. **Defended-general assault.** 12 draws remain out of 120. Now that a path
   exists, breaking a general sitting on 250+ army needs mass arriving
   together, not one stack at a time — `_gather` exists but is unproven, and
   probably needs to rally *near the target* rather than on the spearhead.
3. **Castle placement strategy.** Sites are chosen purely by cheapest price.
   Frontier castles project defense; corner clusters pay the surcharge.
4. **Split moves.** The `split` field is still never used. It allows expanding
   and holding a stack simultaneously.
5. **Threat-aware building.** `BUILD_SAFE_DISTANCE` is a flat 3 cells; it
   ignores how much army the nearby enemy actually has.
6. **Opening book.** The first ~50 turns are pure expander greed.
7. **Why `expander_python` still wins 2 of 40** against `my_bot9`, when
   `my_bot6` swept it 12–0. Worth a look; possibly permissive routing wasting
   turns on real mountains early, or an undefended general.

## Measurement discipline (learned the hard way)

- Score against a **fixed** opponent, not the previous generation: two similar
  bots draw and hide real differences.
- **120 matches minimum** (60 seeds × 2 seats). `my_bot7` looked like a win at
  30 seeds and evaporated at 60.
- **Replicate on a disjoint seed range** (`--seed-start 100`) before believing
  anything.
- Read the score line, not the win count: draws count half and the 95% interval
  tells you whether to care.
- When a generation shows no effect, **instrument a failing game** before
  writing the next one. `my_bot9` came from one debug line, after two
  well-reasoned strategies changed nothing.

## Cost check

`my_bot6` decisions measured over a full 683-turn match: median **0.2 ms**,
worst **0.7 ms**. The per-turn BFS and the O(cells × structures) build-cost scan
are not close to any plausible time budget.

```bash
python competition/matchup2.py competition/agents/my_bot6/run.sh \
  competition/agents/my_bot3/run.sh --mode competition --log t.log
grep '^<- .*p0' t.log | sed 's/.*(\(.*\) ms)/\1/' | sort -rn | head -3
```
