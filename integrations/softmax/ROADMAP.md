# Softmax roadmap

Updated 2026-09-17. Matej has now requested FFA and castle-building leagues
under the existing Coworld, authorizing parallel implementation and deployment.
Both modes are now uploaded and certified as v0.3.0 with full-length hosted
verification. Creating their separate leagues is blocked by a team-only
permission to set the default variant. See [VARIANTS.md](VARIANTS.md) for the
exact blocker, replay links, and Richard handoff.
The original ordering below is retained for context. The order below is a rough estimate of effort, not a
commitment to dates.

## Context and direction

Matej shared his September 15–16 conversation with Richard Higgins (Softmax).
Richard suggested a league/division for bot submissions, leaderboard matches,
and browser replays, then shared a screenshot of a running competition. He
subsequently proposed supporting classic 1v1, FFA, and castle-building games,
collecting remaining usability problems, and improving presentation. Softmax
offered a possible `generals.softmax.com` or `softmax.com/generals` entry point;
the address has not been chosen or confirmed here.

The September 17 checks confirmed that Coworld v0.2.5 is certified and canonical
and that `origin/softmax` is at `0309168`. Richard's open
[PR #141](https://github.com/strakam/generals-bots/pull/141) references round #34
of an existing Generals Competition league.

Read-only API checks on September 17 confirmed the existing
[Generals Competition league](https://softmax.com/observatory/v2?detail=league:league_8c189954-be68-479c-a092-eeb79c436d12)
is public, enabled, and linked to our v0.2.5 Coworld. Its Competition division is
`div_5ee4b276-f330-42e8-b8e4-a6097c779d99`; configuration specifies Elo ranking
and a ten-minute round interval. The league-locks endpoint denies Matej's
account with an owner-only 403, and league-seed administration returns a
Softmax-team-only 403. The individual league owner has not been identified.
Prefer reusing this league; clarify management access with Richard before
planning settings changes or creating a duplicate.

Keep integration work on `softmax`, leaving `master` alone. Preserve classic
capture-only play as a separate option when adding other rulesets. Avoid opening
Matej's browser unless he asks. Do not send feedback to Richard without Matej's
authorization.

## Checklist: roughly easiest to hardest

### 1. Locate and document the existing league — small

- [ ] Find the league and division URLs, owner/admin access, and current game version.
      URLs/version and current access restrictions are confirmed above; ownership
      and how Matej can obtain management access remain to be clarified.
- [ ] Check existing entrants and whether Matej already has a submitted bot.
- [ ] Save the links and a short explanation of joining, submitting, and viewing results.

Done when Matej has one clear starting link and knows what is already running.

### 2. Review Richard's longer-match change — small to medium

- [ ] Review PR #141, which raises the competition cap from 1,200 to 2,000 turns
      while keeping human games at 1,200.
- [ ] Update the replay viewer's current 1,201-frame limit so longer games load.
- [ ] Add meaningful coverage for play beyond turn 1,200 and a full-length replay.
- [ ] Reproduce and diagnose the two WebSocket test failures reported in the PR.
- [ ] Prepare the reviewed change for integration, then build and certify v0.2.6.
- [ ] Publish the verified release and separately update the existing league's
      game version/configuration when proceeding with the rollout.

Done when the longer competition matches and their replays work on Softmax;
merging alone does not update the hosted league.

### 3. Exercise the complete bot submission experience — medium

- [x] Start with an existing repository bot; inspect existing submissions before
      creating another policy or entry. Hunter was submitted on September 17
      under Matej Straka; Softmax reports active competition membership and
      champion status. See [HUNTER.md](HUNTER.md) for the bot link and evidence.
      Two hosted verification matches against Expander completed with Hunter
      wins and zero timeouts. Scheduled league results/standing remain to check.
- [ ] Follow submission through a league match, scores, leaderboard, and browser replay.
- [ ] Check human-versus-bot play as a separate flow, using a fresh lobby when needed.
- [ ] Record confusing steps and reproducible failures. Fix repository problems;
      prepare a short report for platform problems, for Matej to review.

Done when Matej can follow a short set of instructions from bot to visible result.
Building a stronger competitive bot is a separate, optional project.

### 4. Make the game easier to find and use — medium; some Softmax dependencies

- [ ] Use the previous walkthrough to prioritize descriptions, rules, navigation,
      replay visibility, and other presentation fixes.
- [ ] Reuse the existing game graphics and UI assets.
- [ ] Coordinate with Richard on the offered Generals address and its destination.
      Softmax needs to provide the platform routing; don't assume we control it.
- [ ] Verify the agreed public entry point and the route from it to play or submission.

Done when newcomers have a clear entry point and can understand how to participate.

### 5. Add castle-building as a separate mode — medium to large

- [x] Reuse and inspect the existing castle-building engine/modifier code.
- [x] Define the mode's rules, including neutral castles and whether Deathtouch
      belongs in it; enabling castle building need not enable every old modifier.
- [x] Extend configuration, action validation, browser controls, bot documentation,
      and replays for the mode while preserving classic 1v1 behavior.
- [ ] Test and certify it, then choose how to expose it in the existing league/divisions
      or a separate competition based on Softmax's supported configuration.

Done when users can deliberately choose classic or castle-building play and the
rules shown match what the engine executes.

### 6. Add FFA — largest; scope before implementing

- [x] Inspect existing multiplayer work/branches for reusable engine support.
- [x] Define player count, spawning, elimination, captured territory, ranking,
      scoring, disconnect behavior, and fog of war.
- [x] Check Softmax's support for the intended multi-player league and lobby setup.
- [x] Extend the adapter's current two-player assumptions in authentication,
      observations, scheduling, results, bots, colors, scoreboard, and replays.
- [ ] Test multi-player outcomes and information boundaries, certify, and expose
      FFA as a separate mode.

Done when full FFA matches, rankings, and replays work end to end. This is more
than increasing a player-count setting in the current 1v1 adapter.

## Current step

FFA and castle building are implemented, tested, and published as v0.3.0.
Ask Richard to create their variant-selected leagues (the API restricts that
field to Softmax team members), then verify division settings and scheduled rounds. Classic rules and
entrants remain available. Ownership access is now granted and classic league
rounds are running, superseding the earlier access-blocked notes above.
PR #141 remains a separate longer-match change; it is not merged by this work.
