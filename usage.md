# Tooling — what to run

Six tools live in `competition/`. All commands are run **from the repo root**.

| Tool | What it's for |
|---|---|
| [`matchup.py`](competition/matchup.py) | Original bot-vs-bot runner. Kept as-is for reference. |
| [`matchup2.py`](competition/matchup2.py) | Bot vs bot, correct ruleset, transcripts and recordings. **Use this one.** |
| [`matchup3.py`](competition/matchup3.py) | Human vs bot in a browser; also the laptop-side worker. |
| [`relay.py`](competition/relay.py) | Public front end on a VPS. Serves the site, brokers to a worker. |
| [`arena.py`](competition/arena.py) | Bulk head-to-head evaluation with win rates and confidence intervals. |
| [`replay.py`](competition/replay.py) | Watch a saved `.rpl` in the pygame GUI. |

Agents are directories under `competition/agents/<name>/` containing `run.sh`.
Anything with a `run.sh` is a valid agent; tools take either a bare name
(`my_bot9`) or a path to the script, depending on the tool.

---

## matchup2.py — bot vs bot

The workhorse. Same engine path everything else uses: build-castles and
deathtouch applied, per-seed rectangular boards.

```bash
# competition ruleset (what you almost always want)
python competition/matchup2.py \
  competition/agents/my_bot9/run.sh \
  competition/agents/expander_python/run.sh \
  --mode competition

# with a transcript, a saved replay, and the GUI
python competition/matchup2.py <a0> <a1> --mode competition \
  --log match.log --save match.rpl --gui
```

| Flag | Meaning |
|---|---|
| `--mode competition` | Pins the whole official ruleset. Overrides the three flags below. |
| `--grid-size N` `--truncation N` `--perfect-info` | Manual ruleset, for quick small-board debugging. |
| `--seed N` | Board, terrain and general positions. **Defaults to 0**, so every run is the same map unless you change it. |
| `--log PATH` | Transcript of the engine↔agent conversation, with per-decision timing. |
| `--show-grids` | Adds the three HxW grids per turn to `--log` (~2 MB/match). |
| `--save PATH` | Compact replay (~2 KB), viewable with `replay.py`. |
| `--gui` `--fps N` `--cell-size PX` | Play headless, then open a scrubbable window. Cell size auto-fits your screen. |

Reading a transcript — `->` is engine-to-agent, `<-` is the agent's reply:

```bash
grep '^<- .*p0' match.log | tail -20          # your bot's last 20 moves
grep -c '^<- .*p0.*: 2 ' match.log            # how many builds it attempted
grep '^<- .*p0' match.log | sed 's/.*(\(.*\) ms)/\1/' | sort -rn | head -3
```

`matchup.py` takes the same flags minus `--log/--save/--cell-size`, but it
bypasses the ruleset modifiers — a build action silently executes as a *move*,
and deathtouch never fires. Prefer `matchup2.py`.

---

## arena.py — bulk evaluation

Plays every seed in **both seat orders** so seat advantage cancels, then reports
win/loss/draw, a score (draws count half) and a 95% interval.

```bash
python competition/arena.py my_bot9 my_bot6 --seeds 60 --jobs 8

# independent second sample — same bots, disjoint maps
python competition/arena.py my_bot9 my_bot6 --seeds 60 --seed-start 100 --jobs 8

# archive every match for later viewing
python competition/arena.py my_bot9 my_bot6 --seeds 60 --jobs 8 --save-dir replays/
```

| Flag | Meaning |
|---|---|
| `--seeds N` | Number of seeds; matches = 2 × N. |
| `--seed-start N` | First seed. Use it for a fresh, non-overlapping map set. |
| `--jobs N` | Parallel workers. ~0.4 s per match per worker. |
| `--grid-size N` | Plain NxN board instead of `--mode competition`. |
| `--save-dir DIR` | One `.rpl` per match (~2 KB each). |
| `--verbose` / `--quiet` | Per-match lines / totals only. Default is a progress bar with ETA. |

**Read the score line, not the win count.** Rough sizing: 120 matches ≈ 40 s at
`--jobs 8`. Anything under 120 matches cannot separate two similar bots — see
the "measurement discipline" section of [my_bot.md](my_bot.md).

---

## replay.py — watch a saved match

```bash
python competition/replay.py match.rpl                    # auto-fit window
python competition/replay.py match.rpl --fps 20 --cell-size 24
python competition/replay.py match.rpl --info             # header only, no window
```

Controls are drawn in the window: SPACE play/pause, ←/→ (or H/L) step a frame
and hold to run, R restart, Q quit.

A `.rpl` stores only the seed and the action stream, so playback **re-simulates**
— a 1200-turn match is ~10 KB but takes a second or two to open. It is therefore
only valid against the engine that produced it: change the rules in
`generals/core/` and old files rebuild into a different game.

Files come from `matchup2.py --save`, `arena.py --save-dir`, or the browser's
**Download .rpl** button.

---

## matchup3.py — human vs bot in a browser

Serves the page and the game API from one process. Bots run over the ordinary
stdio protocol, unmodified.

```bash
# everything local — this is all you need to play
python competition/matchup3.py --port 8080
# then open http://localhost:8080/
```

Pick the opponent in the **Bot** dropdown; every agent with a `run.sh` is
offered. In-browser controls: click selects, double-click selects with a
half-army move armed, **right-click builds a castle**, arrows/WASD queue moves,
`Q` clears the queue, `Z` toggles half, `Esc` deselects. **Autopilot** hands your
seat to a bot and can be toggled back mid-game. **Show replay** scrubs finished
matches, from either seat or with no fog, and exports `.rpl`.

| Flag | Meaning |
|---|---|
| `--host` `--port` | Bind address. `0.0.0.0` to expose it; default is loopback only. |
| `--replays N` | Finished matches kept for replay (default 10, ~100 KB each). |
| `--relay URL` `--token S` | Worker mode — see below. |
| `--cors ORIGIN` | Only if the frontend is hosted elsewhere (e.g. Vercel). |

---

## relay.py — public site, bots on your laptop

Two processes when the engine must **not** run on the VPS. The laptop dials out,
so nothing needs to be reachable behind your NAT. The relay holds no engine at
all — stdlib only, so the VPS needs no `pip install`.

```bash
# on the VPS
python competition/relay.py --host 0.0.0.0 --port 8080 --token SECRET --replays 10

# on your laptop (engine + bots live here)
python competition/matchup3.py --relay https://your-domain --token SECRET
```

Locally, to rehearse the split, point the worker at `http://127.0.0.1:8080`.
Both take `--token` and they must match, or the worker gets `403` and the site
shows "no bots live". `--relay` wants the **base URL**, not `/agent/sync`.
Expect ~10 s before bots appear — that is JAX importing on the worker.

Use `-u` on both: without it Python block-buffers stdout when it isn't a
terminal and the logs look dead when everything is fine.

Health check:

```bash
curl -s localhost:8080/api/bots
# {"bots": ["expander_python", ..., "my_bot9"], "live": true}
```

Deployment files: [deploy/matchup3.service](deploy/matchup3.service) (systemd)
and [deploy/Caddyfile](deploy/Caddyfile) (HTTPS via Let's Encrypt). On Oracle
Cloud, opening a port takes **two** steps — a VCN Security List ingress rule
*and* the instance firewall. Missing the first is the usual reason it looks dead.

---

## Current limits

- `matchup3`/`relay` hold **one match and one worker** at a time; a second
  visitor takes over the running game.
- The perspective toggle only unlocks once a match ends — streaming the true
  board to a live player would hand them the answer.
- `arena.py` results below ~120 matches are noise.
