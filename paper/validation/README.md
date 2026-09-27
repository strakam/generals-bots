# Does the engine reproduce real generals.io games?

Validation against ranked 1v1 replays of the current game (replay formats 16 and 18, 2026). The replays, the
site's own engine run under Node for ground truth, the pipeline and the results live in the paper repository
under `review/replays_2026/`; this directory holds the engine-side tools.

* `gior.py` decodes a `.gior` replay (the format served from the public replay buckets) following
  `Replay.deserialize` of the client bundle.
* `replay_prep.py` turns a decoded replay into engine inputs (padded grid, per-tick actions); base game only.
* `replay_recent.py 'dir/*.gior' [--no-trade] [--out f]` plays every replay through `game.step` as one
  `lax.scan` and checks, from the record alone, that no recorded move is rejected and that each game ends the
  way the record says (capture by the recorded winner on the recorded turn; or no early end for a surrender).
* `board_diff.py <siteboards> 'dir/*.gior' [--out f]` compares our board with the site engine's board tile by
  tile at every tick.

Result (2026-09-20, 10,000 base-game replays drawn at random, 6.3 million moves): boards identical to the site's
engine at every in-game tick in all 10,000 games; 0 recorded moves rejected.
