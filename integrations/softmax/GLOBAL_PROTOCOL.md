# Generals Coworld global and replay protocol v1

`GET /client/global` serves the public score viewer. Connect to `/global` for
JSON text updates. No spectator credential is required, so **this stream must
be safe for either competing player to read**. Incoming messages never mutate
the game. WebSocket ping receives a matching pong.

Updates contain `type: "global"`, `protocol_version: 1`, `phase`, `turn`,
`max_turns`, two or four `players` display names, per-slot `army`, `land`, and `eliminated` arrays,
`ruleset`, `result`, and `board`. During the match, `board` and `result` are null. Phases
are `waiting`, `playing`, `resolving`, `saving`, `finished`, and `failed`;
brief transitional phases need not each be delivered to a slow viewer.

Only `finished` reveals a board and result. `GET /replay.json` returns HTTP 409
until then. The game server exits after completing the hosted episode; hosted
replay viewing uses the static bundle, independent of the game container.

The replay is JSON:

- `format: "generals-coworld"`, `version: 1`, `ruleset: "classic"` or `"build_castles"`.
  Archived release 0.1.0 replays use `ruleset: "competition"` instead.
- Actual `seed`, `height`, `width`, and two or four display names in `players`.
- `frames`: initial frame, then one frame after every applied turn. Each has
  `turn`, `type_grid`, `owner_grid`, `army_grid`, `army`, `land`, and `eliminated`.
- Replay owners are **absolute**: 0 neutral, otherwise slot + 1. Replay types
  use the same integer codes as player observations but contain no fog.
- `turns`: each attempted turn's `turn`, `actions` and Boolean `timed_out` arrays (one entry per slot), `forfeited` (the eliminated slot indices for this attempt), and `applied`.
  Apply forfeits before moves when resimulating. A terminal forfeit stops before
  the move/growth tick (`applied: false`); a final frame records its elimination
  state at the same tick. Nonterminal FFA forfeits allow the other moves to run.
- `result`: the same final scoring object sent to players.

There are no tokens or presigned upload URLs in replays. Board frames allow
browser playback without resimulating JAX. Seed and action history additionally
allow deterministic resimulation against the recorded ruleset release.

The static viewer loads a replay URL from `#replay=`, falling back to `?replay=`.
It accepts raw JSON or gzip identified by magic bytes. Internal assets use
relative URLs and replay fetching needs CORS when hosted on a different origin.
The viewer sends `coworld-replay` loading/phase/ready/error messages to its
embedding parent, autoplays, loops, and exposes pause, seek, speed and restart.
