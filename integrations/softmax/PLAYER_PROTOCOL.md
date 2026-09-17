# Generals Coworld player protocol v1

Connect to the runner-provided `COWORLD_PLAYER_WS_URL`, a fully formed URL like
`ws://game:8080/player?slot=0&token=...`. Never log the URL/token. Use WebSocket
JSON text messages. Python `websockets` clients should use `ping_timeout=None`.
The game answers WebSocket ping frames normally. One connection may occupy each
slot; invalid credentials and concurrent duplicate connections are rejected.

The game sends a `hello` with `protocol_version: 1`, your `slot` (0 through player count minus one), board
`height` and `width`, two or four display names in `players`, and
`ruleset: "classic"` or `"build_castles"`. Wait for an `observation` before sending actions.

Each observation has:

| Field | Meaning |
| --- | --- |
| `type` | `"observation"` |
| `protocol_version`, `slot`, `height`, `width`, `players` | Same as the handshake |
| `turn` | Current state tick, starting at 0 |
| `last_move_executed` | Whether your preceding turn's move executed; `null` initially or after pass/build. An executed attack need not capture its destination. |
| `max_turns` | Episode cap; 1200 in the competition variant |
| `turn_timeout_seconds` | Deadline duration from server observation publication |
| `my_land`, `my_army`, `opp_land`, `opp_army` | Public totals; opponents aggregate all other players |
| `public_scores` | Per-slot `army`, `land`, and `eliminated` arrays |
| `eliminated` | Whether your slot is eliminated; stop sending actions and await final |
| `ruleset` | Same as handshake |
| `visible_owner_grid` | Fog-masked absolute owners: 0 neutral/unknown, otherwise slot + 1 |
| `build_cost_grid` | Building mode only: cost on owned plain tiles, otherwise 0 |
| `last_build_executed` | Whether preceding build succeeded; null after other actions or initially |
| `type_grid` | H rows of W integers: 0 fog, 1 plain, 2 mountain, 3 castle, 4 general, 5 structure in fog |
| `owner_grid` | H×W, perspective-relative: 0 neutral/unknown, 1 me, 2 any opponent |
| `army_grid` | H×W visible army counts; zero in fog |

Reply with exactly these three fields:

```json
{"type":"action","turn":12,"action":[0,5,8,3,0]}
```

`action` contains five integers `[kind, row, col, direction, split]`:

- `kind`: 0 move, 1 pass, or 2 build only in `build_castles` mode. Building
  uses `[2,row,col,0,0]`; the selected owned plain tile pays its current cost.
  Classic games reject kind 2 without consuming the action slot.
- `row`, `col`: a source cell within the board bounds.
- `direction`: 0 up, 1 down, 2 left, 3 right.
- `split`: 0 all-but-one, 1 half (rounded down).

Use `[1,0,0,0,0]` to pass. All five values must
be valid integers even when a field is unused. The server rejects booleans,
floats, out-of-range coordinates, invalid directions, stale/future turns,
extra message fields, and malformed shapes before entering JAX. Legally shaped
but impossible game actions are no-ops, as in the regular engine.

Only the **first valid message for the current turn** is accepted. Actions from
all surviving players resolve together using the existing engine's move-order rules.
The server has no action queue: act only on the newest observation. The browser
player stores premoves locally and submits one action per observation using this
same protocol. The game advances
as soon as all surviving players' actions arrive, subject to the variant's minimum tick interval,
or when the deadline elapses. The human variant's minimum interval is 500 ms,
with a one-second action deadline to allow for proxy transport. Browser clients
send idle passes early; queued inputs after that pass execute on the next tick.
The competitive bot variant has no artificial delay and retains a 500 ms deadline.

Protocol errors return `{"type":"error","message":"..."}` and do not fill
the action slot. You may correct the action before the deadline. Oversized
messages close the connection. A missed action becomes a pass; 20 consecutive
misses forfeit. Reconnection resumes with `hello` and, if an action window is
open, its current observation. Accepted actions and timeout counters survive
reconnection. Forfeits eliminate the missing players. In FFA, other players continue;
forfeited territory becomes neutral and their general becomes a castle. One
survivor wins and no survivors draw. Eliminated slots no longer have deadlines. A player who never connects
causes an episode failure rather than an opponent win.

The final message is `{"type":"final","result":{...}}`, followed by a clean
close. `result.scores` is one number per absolute slot (+1 for the winner, -1 for all others, or all zeros on a draw), `winner`
is an absolute slot or -1 for a draw, `reason` is `general_capture`, `turn_limit`, `forfeit`,
or `double_forfeit`, `turns` is the final state tick, and `army`, `land`, and
`timeouts` have one entry per slot. Exit cleanly after receiving the final message.
`{"type":"failure","result":null}` means the game failed to complete.

No map seed, hidden board, other player's token, or opponent action is provided
during play. The completed public replay reveals the full match. The baseline
bridge in `player.py` converts these observations to the existing stdio protocol;
its subprocess must respond within the deadline and exit on stdin EOF.

The stdio bridge preserves its relative-owner format in FFA (all enemies are 2).
Bots that need individual opponent identities or build prices should use the
JSON protocol. No other player's hidden tiles are revealed by the added fields.
