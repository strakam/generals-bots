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
records and public league links are added below after verification.
