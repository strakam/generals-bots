# Softmax integration and player work

This directory contains the game adapter and a participant example. Read
`README.md` for the game contract, `ROADMAP.md` for the agreed scope, and
`HUNTER.md` before working on the Hunter participant. Keep work on the `softmax`
branch. Do not open or navigate the user's browser unless requested; use an
isolated headless browser for checks.

For player participation, read the league's current guide:
https://softmax.com/api/observatory/v2/participate?league_id=league_8c189954-be68-479c-a092-eeb79c436d12

Supporting documentation:

- https://docs.softmax.com/coworld/overview
- https://docs.softmax.com/coworld/cli
- https://docs.softmax.com/coworld/build-a-player/debug-hosted-episodes

Apply the participant workflow to player work, not to unrelated game-adapter
maintenance. User instructions and existing authorization take precedence.

- The current objective is testing participation with the existing Hunter.
  Its v1 policy is already uploaded and entered. Inspect its recorded IDs and
  status before creating another policy, submission, or competition.
- The user's goal and constraints are already recorded in `HUNTER.md`. Do not
  start an open-ended strategy optimization loop from a request to inspect
  participation or replay links.
- If the user requests optimization, use comparable hosted Experience Request
  batches, inspect results/logs/replays, and propose one focused change. Obtain
  approval for strategy changes outside the user's existing authorization.
  Local matches are protocol/packaging checks, not strength evidence.
- Keep upload, hosted tests, league placement, and actual league rounds distinct
  when reporting results. Two wins over Expander do not establish leaderboard
  strength. An active membership alone does not prove a league match has run.
- Update the participant runbook with versions, commands, evidence, and browser
  links. Verify the intended episode, not just the presence of a replay iframe.
- Preserve evidence of platform/documentation problems for a concrete report.
  Do not post issues or messages to others without the user's authorization.
- Keep tokens and credential-bearing connection URLs out of logs and commits.
