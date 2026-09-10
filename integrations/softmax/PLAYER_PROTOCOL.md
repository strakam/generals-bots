# Generals Coworld player protocol v1

Connect to the runner-provided `COWORLD_PLAYER_WS_URL`, a fully formed URL like
`ws://game:8080/player?slot=0&token=...`. Never log the URL/token. Use WebSocket
JSON text messages. Python `websockets` clients should use `ping_timeout=None`.
The game answers WebSocket ping frames normally. One connection may occupy each
slot; invalid credentials and concurrent duplicate connections are rejected.

The game sends a `hello` with `protocol_version: 1`, your `slot` (0 or 1), board
`height` and `width`, the two display names in `players`, and
`ruleset: "competition"`. Wait for an `observation` before sending actions.

Each observation has:

| Field | Meaning |
| --- | --- |
| `type` | `"observation"` |
| `protocol_version`, `slot`, `height`, `width`, `players` | Same as the handshake |
| `turn` | Current state tick, starting at 0 |
| `max_turns` | Episode cap; 1200 in the competition variant |
| `turn_timeout_seconds` | Deadline duration from server observation publication |
| `my_land`, `my_army`, `opp_land`, `opp_army` | Public totals |
| `type_grid` | H rows of W integers: 0 fog, 1 plain, 2 mountain, 3 castle, 4 general, 5 structure in fog |
| `owner_grid` | H×W, perspective-relative: 0 neutral/unknown, 1 me, 2 opponent |
| `army_grid` | H×W visible army counts; zero in fog |

Reply with exactly these three fields:

```json
{"type":"action","turn":12,"action":[0,5,8,3,0]}
```

`action` contains five integers `[kind, row, col, direction, split]`:

- `kind`: 0 move, 1 pass, 2 build castle.
- `row`, `col`: a source cell within the board bounds.
- `direction`: 0 up, 1 down, 2 left, 3 right.
- `split`: 0 all-but-one, 1 half (rounded down).

Use `[1,0,0,0,0]` to pass and `[2,row,col,0,0]` to build. All five values must
be valid integers even when a field is unused. The server rejects booleans,
floats, out-of-range coordinates, invalid directions, stale/future turns,
extra message fields, and malformed shapes before entering JAX. Legally shaped
but impossible game actions are no-ops, as in the competition engine.

Only the **first valid message for the current turn** is accepted. Actions from
both players resolve together using the existing engine's move-order rules.
There is no action queue: act only on the newest observation. The game advances
as soon as both actions arrive, subject to the variant's minimum tick interval,
or when the deadline elapses. The human variant's minimum interval is 500 ms;
the competitive bot variant has no artificial delay.

Protocol errors return `{"type":"error","message":"..."}` and do not fill
the action slot. You may correct the action before the deadline. Oversized
messages close the connection. A missed action becomes a pass; 20 consecutive
misses forfeit. Reconnection resumes with `hello` and, if an action window is
open, its current observation. Accepted actions and timeout counters survive
reconnection. Two simultaneous forfeits draw. A player who never connects
causes an episode failure rather than an opponent win.

The final message is `{"type":"final","result":{...}}`, followed by a clean
close. `result.scores` is one number per absolute slot (+1/-1 or 0/0), `winner`
is 0/1 or -1 for a draw, `reason` is `general_capture`, `turn_limit`, `forfeit`,
or `double_forfeit`, `turns` is the final state tick, and `army`, `land`, and
`timeouts` are two-element arrays. Exit cleanly after receiving the final message.
`{"type":"failure","result":null}` means the game failed to complete.

No map seed, hidden board, other player's token, or opponent action is provided
during play. The completed public replay reveals the full match. The baseline
bridge in `player.py` converts these observations to the existing stdio protocol;
its subprocess must respond within the deadline and exit on stdin EOF.
