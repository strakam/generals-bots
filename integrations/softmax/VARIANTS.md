# Additional Generals leagues

The September 17 request prioritizes Richard's proposed FFA and castle-building
leagues under the existing Generals Coworld. Work stays on `softmax`.

## Modes in release 0.3.0

| Variant | Seats | Rules |
| --- | --- | --- |
| `competition` | 2 | Existing classic rules, neutral castles |
| `ffa` | 4 | Classic combat, last surviving general wins |
| `castles` | 2 | Build castles on owned plain land; no starting neutral castles |
| `human`, `ffa-human`, `castles-human` | 2, 4, 2 | Corresponding rules, 500 ms minimum tick and 1 s deadline |

All modes retain a 1,200-turn cap and disable Deathtouch. Richard's separate
PR #141 for 2,000-turn classic competition games is not included in this release.
FFA uses +1 for the winner, -1 for every other player, and all zeros at the turn
cap; it does not yet award intermediate finishing ranks. See README and the
player protocol for capture, timeout, building-cost, and fog rules.

## Platform setup

Publish one certified version of `generals-competition` containing all variants.
Create one league seed per additional mode, pointing its `default_variant_id`
to `ffa` or `castles`. Use Softmax's platform commissioner, then configure a
competition division and ladder settings. A seed alone is not a running ladder.
Keep the original classic league and its entrants.

The September 17 live API now reports Matej as an owner of the original league;
settings and locks are accessible. Earlier owner-only 403s no longer apply.
Listing all Coworld league seeds is still restricted to the Softmax team, which
does not establish whether creating a seed for an owned Coworld is denied.
The original league follows the canonical Coworld version (game version unlocked).

## Validation and maintenance

The same engine, server, player bridge and replay renderer implement all modes.
New rules or visuals require one Coworld build; league scheduler configuration
is managed separately. Add focused checks when changing elimination, fog,
build receipts, player counts, or browser queues. Certify the container and run
hosted verification for each affected mode; classic-only certification does not
prove FFA or building works.

Local smoke/full matches:

```bash
python -m integrations.softmax.local --variant ffa --seed 7
python -m integrations.softmax.local --variant castles --seed 7
GENERALS_BROWSER_TESTS=1 pytest tests -q
```

Both 1,200-turn local matches completed with zero player timeouts on September
17. The building replay contained nine construction actions and nine castles;
the FFA replay had four independent player ownership colors. Hosted deployment
records are below. These baseline matches validate integration, not playing
strength or draw-rate balance.

## Published September 17, 2026

- Canonical and certified: **generals-competition 0.3.0**, Coworld
  `cow_8c336ec5-413c-4cfc-99b5-fbe2c1f71b5d`.
- Release source: `7632640` on `softmax` (implementation `3af20af`). Master
  remains `e596ffe5cc7e065d3698b9b7d6475a1455d517d2`.
- 188 tests passed, including real Chromium checks. All three local container
  certification fixtures passed their ten checks. Both 1,201-frame new-mode
  replays were checked through their final frame in Chromium.
- Hosted certification passed ten checks and five classic smoke episodes.
- Two full-length hosted episodes per new mode completed at turn 1,200 with
  zero timeouts and valid results/replays. These four baseline games drew.
  Building episodes contained eight and five construction actions.
- Public hosted verification requests: FFA
  `xreq_18150375-c866-4f82-a766-cc7a885bd79a`, castles
  `xreq_2d7c9001-a5cc-46a1-bb5d-ab2dbf817014`.
- Baselines: `strakam-generals-expander:v2`
  (`c2c6eb89-51ff-4fba-928d-e35dfa278357`) and
  `strakam-generals-builder:v1` (`0d7a26da-c81a-4173-906e-ee1fff21ffc4`).
  Both reuse the uploaded lightweight player image; no new league submissions
  were made. Existing Hunter participation remains unchanged.

Website replays:

- [Four-player FFA](https://softmax.com/generals-competition?e=ca3ebdc5-b9e5-413e-8169-7df02c54f692)
- [Build your own castles](https://softmax.com/generals-competition?e=933f9bbf-9338-4f9f-b00d-4a0d7d574079)

Both replay links were checked anonymously on the Softmax website, including
the exact episode iframe, variant title, player colors and final frame. The
outer standings still belong to classic; these are verification episodes, not
FFA/castles league results.

The original classic league now follows v0.3.0 automatically; its selected
`competition` variant and existing membership are unchanged. Both new modes
are uploaded variants, **not yet separate running leagues**.

## Remaining Softmax permission blocker

The valid seed-creation request for FFA returned HTTP 403:
`Only Softmax team members may set the default variant`.
This restriction is separate from ownership of the existing league. Do not
retry around the access check or create an incorrectly configured classic
league as a substitute. The failed request created no league; a subsequent
GET confirmed that only the existing Generals Competition league is present.

The seed `overrides` schema also rejects `public`, `hidden`, and `short_name`
with HTTP 422, even though those fields exist in league responses. Use only
supported seed override fields. These endpoints are absent from the public
OpenAPI, so league administration needs better documentation.

A Softmax team member can start with these installed-SDK commands:

```bash
coworld league create generals-competition ffa "Generals FFA" \
  --default-variant ffa --set commissioner_key=platform
coworld league create generals-competition castles "Generals Build Your Castles" \
  --default-variant castles --set commissioner_key=platform
```

Then declare a competition division (`PUT /v2/leagues/{id}/divisions`) and enable
its ladder (`POST /v2/leagues/{id}/settings`). The endpoint paths are documented
by the SDK, but exact request wrappers have not been verified. Reuse classic's
ten-minute schedule, Elo ranking and per-user entrant limit; FFA needs four
seats (a balanced-rotation scheduler), castles two (Swiss-neighbor works for
classic). Verify the actual settings and submit tested baselines before claiming
scheduled rounds work. Preserve classic as the default website league if
supported: the game's `default_league_id` is currently null, and the frontend
otherwise chooses the first listed league.

Suggested message for Richard (not sent):

> FFA (4 players) and build-your-own-castles (1v1, no Deathtouch) are uploaded
> and certified in Generals v0.3.0. Both passed full-length hosted matches.
> Could you create their leagues under the existing Coworld, using variants
> `ffa` and `castles`? My account's seed-creation request gets “Only Softmax
> team members may set the default variant.” Please keep classic 1v1 available.

Ignored deployment logs, replay artifacts and exact API responses are kept in
`local-output/release-0.3.0/`; browser evidence is in
`local-output/variant-browser-validation/`.
