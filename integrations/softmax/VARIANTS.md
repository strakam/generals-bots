# Generals modes and leagues

One `generals-competition` Coworld carries every mode; each bot mode runs as its
own Softmax league. Work stays on `softmax`.

## Modes in release 0.3.2

| Variant | Seats | Rules | Turn cap |
| --- | --- | --- | --- |
| `competition` | 2 | Classic rules, neutral castles | 2,000 |
| `ffa` | 4 | Classic combat, last surviving general wins | 2,000 |
| `castles` | 2 | Build castles on owned plain land; no starting neutral castles | 2,000 |
| `human`, `ffa-human`, `castles-human` | 2, 4, 2 | Corresponding rules, 500 ms minimum tick and 1 s deadline | 1,200 |

Human variants keep 1,200 turns: at two turns per second, 2,000 turns would take
at least 17 minutes and risk Softmax's 20-minute hosted episode deadline. Every
mode disables Deathtouch and resolves moves in generals.io's order (see README).
FFA uses +1 for the winner, -1 for every other player, and all zeros at the turn
cap; it does not yet award intermediate finishing ranks. See README and the
player protocol for capture, timeout, building-cost, and fog rules.

## Leagues

| League | Variant | League ID | Division |
| --- | --- | --- | --- |
| [Classic 1v1](https://softmax.com/observatory/v2?detail=league:league_8c189954-be68-479c-a092-eeb79c436d12) | `competition` | `league_8c189954-be68-479c-a092-eeb79c436d12` | Competition `div_5ee4b276-f330-42e8-b8e4-a6097c779d99` |
| [FFA 1v1v1v1](https://softmax.com/observatory/v2?detail=league:league_f189ad01-2700-4fc8-ba89-f9de477099e4) | `ffa` | `league_f189ad01-2700-4fc8-ba89-f9de477099e4` | FFA `div_ac86d8a0-7b43-41bb-b6e7-1867f7b47d08` |
| [Build 1v1](https://softmax.com/observatory/v2?detail=league:league_b371d42e-bed5-4f33-ac02-fca229860ddf) | `castles` | `league_b371d42e-bed5-4f33-ac02-fca229860ddf` | Castles `div_ede858f5-6c57-4d03-b16d-a843a35e8bc2` |

All three are public, use the platform commissioner with Elo ranking and the
Swiss-neighbor scheduler, and follow the canonical Coworld version: a certified
upload moves every league to it without league changes. On September 27 the
classic league ran rounds every 30 minutes and the other two every 10.

The FFA and Build leagues were created on September 17 (20:24 UTC) by the
Softmax team, after Matej's account was refused with HTTP 403
`Only Softmax team members may set the default variant`. Matej owns the classic
league; setting a league's default variant remains team-only.

## Validation and maintenance

The same engine, server, player bridge and replay renderer implement all modes.
New rules or visuals require one Coworld build; league scheduler configuration
is managed separately. Add focused checks when changing elimination, fog,
build receipts, player counts, or browser queues. Certify the container and run
hosted verification for each affected mode; classic-only certification does not
prove FFA or building works.

Local smoke/full matches (bot games default to the hosted 2,000-turn cap):

```bash
python -m integrations.softmax.local --variant ffa --seed 7
python -m integrations.softmax.local --variant castles --seed 7
GENERALS_BROWSER_TESTS=1 pytest tests -q
```

These baseline matches validate integration, not playing strength or draw-rate
balance. `test_live_public_routes_do_not_reveal_hidden_state` fails in roughly one
run in five: cancellation interrupts the WebSocket cleanup at shutdown. Richard's
Codex proposed shielding that cleanup on PR #141; the fix is not merged.

## Releases

Ignored deployment logs, manifests, replay artifacts and exact API responses are
kept in `local-output/release-<version>/`.

### 0.3.2 — September 27, 2026

- FFA and castle-building bot variants raised to 2,000 turns; human variants
  stay at 1,200. Release source `04e76ba` on `softmax`.
- Coworld `cow_08589716-ac45-4575-b3de-7e4cab635fc8`, canonical; all three leagues
  moved to it. Hosted certification passed ten checks and five smoke episodes.
- Full 2,000-turn FFA and castles games passed locally, in container
  certification and hosted (69 s and 49 s), all with zero timeouts:
  [FFA](https://softmax.com/generals-competition?e=78143037-11fc-4c4c-b6ca-2c62dbd90712),
  [castles](https://softmax.com/generals-competition?e=4a5bd1d4-8638-402b-91b8-228ecff085c9).

### 0.3.1 — September 27, 2026

- Master's replay-verified generals.io rules and faster simulator merged into
  `softmax` (`46c48c4`); general trade stays off, as in master's default env.
- Richard's PR #141 merged (`2f731fe`): classic bot games to 2,000 turns, replay
  viewer up to 2,001 frames. Release source `555622d`.
- Coworld `cow_cc72cd7d-5c39-469e-bae2-d79ecb8aebd3`. 207 tests plus 7 Chromium
  checks passed; local and hosted certification passed. Hosted full-length games
  in every mode completed with zero timeouts:
  [classic, 2,000 turns](https://softmax.com/generals-competition?e=5962de96-e834-48b4-9677-1d20db5722d1),
  [FFA](https://softmax.com/generals-competition?e=b657da12-19be-4bfe-bb59-dca60368d9f7),
  [castles](https://softmax.com/generals-competition?e=0368e1e7-ca3e-455c-8aec-574833f3a9e5).

### 0.3.0 — September 17, 2026

- Added `ffa`, `castles` and their human variants. Coworld
  `cow_8c336ec5-413c-4cfc-99b5-fbe2c1f71b5d`; release source `7632640` on
  `softmax` (implementation `3af20af`).
- 188 tests passed, including real Chromium checks. All three local container
  certification fixtures passed their ten checks. Hosted certification passed
  ten checks and five classic smoke episodes.
- Two full-length hosted episodes per new mode completed at turn 1,200 with
  zero timeouts and valid results/replays; building episodes contained eight
  and five construction actions. Requests: FFA
  `xreq_18150375-c866-4f82-a766-cc7a885bd79a`, castles
  `xreq_2d7c9001-a5cc-46a1-bb5d-ab2dbf817014`. Website replays:
  [FFA](https://softmax.com/generals-competition?e=ca3ebdc5-b9e5-413e-8169-7df02c54f692),
  [castles](https://softmax.com/generals-competition?e=933f9bbf-9338-4f9f-b00d-4a0d7d574079).

Hosted verification uses the baselines `strakam-generals-expander:v2`
(`c2c6eb89-51ff-4fba-928d-e35dfa278357`) and `strakam-generals-builder:v1`
(`0d7a26da-c81a-4173-906e-ee1fff21ffc4`), registered for 0.3.0 on the uploaded
lightweight player image. They are not league entrants.
