# Generals Coworld global and replay protocol v1

`GET /client/global` serves the public score viewer. Connect to `/global` for
JSON text updates. No spectator credential is required, so **this stream must
be safe for either competing player to read**. Incoming messages never mutate
the game. WebSocket ping receives a matching pong.

Updates contain `type: "global"`, `protocol_version: 1`, `phase`, `turn`,
`max_turns`, two `players` display names, two-element `army` and `land` totals,
`result`, and `board`. During the match, `board` and `result` are null. Phases
are `waiting`, `playing`, `resolving`, `saving`, `finished`, and `failed`;
brief transitional phases need not each be delivered to a slow viewer.

Only `finished` reveals a board and result. `GET /replay.json` returns HTTP 409
until then. The game server exits after completing the hosted episode; hosted
replay viewing uses the static bundle, independent of the game container.

The replay is JSON:

- `format: "generals-coworld"`, `version: 1`, `ruleset: "competition"`.
- Actual `seed`, `height`, `width`, and two display names in `players`.
- `frames`: initial frame, then one frame after every applied turn. Each has
  `turn`, `type_grid`, `owner_grid`, `army_grid`, `army`, and `land`.
- Replay owners are **absolute**: 0 neutral, 1 slot 0, 2 slot 1. Replay types
  use the same integer codes as player observations but contain no fog.
- `turns`: each attempted turn's `turn`, two `actions`, two Boolean `timed_out`
  flags, and `applied`. The final attempt that causes a forfeit is not applied
  to the engine; this is explicitly marked `applied: false`.
- `result`: the same final scoring object sent to players.

There are no tokens or presigned upload URLs in replays. Board frames allow
browser playback without resimulating JAX. Seed and action history additionally
allow deterministic resimulation against the recorded ruleset release.

The static viewer loads a replay URL from `#replay=`, falling back to `?replay=`.
It accepts raw JSON or gzip identified by magic bytes. Internal assets use
relative URLs and replay fetching needs CORS when hosted on a different origin.
The viewer sends `coworld-replay` loading/phase/ready/error messages to its
embedding parent, autoplays, loops, and exposes pause, seek, speed and restart.
