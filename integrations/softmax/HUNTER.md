# Hunter on Softmax

Submitted on 2026-09-17 as `strakam-generals-hunter:v1` under Matej Straka's
default player. Softmax reported the submission as `placed`, with membership
`competing` / `active` and `is_champion=true`.

- [Hunter bot page](https://softmax.com/observatory/v2?tab=uploads&detail=policy-version:d6649fb2-6082-4bab-9e3a-7bd28422679f) (sign-in required)
- Policy version: `d6649fb2-6082-4bab-9e3a-7bd28422679f`
- Submission: `sub_a9661105-2bfb-483d-aa64-2f90fbe1473d`
- Membership: `lpm_c980b650-5734-49e0-a786-a6b86f3466e0`

This player runs the existing `generals/agents/hunter_agent.py` unchanged. The
`hunter_player.py` client reconstructs the fog-limited engine observation from
the Coworld protocol and submits Hunter's actions directly over WebSocket.
Both player perspectives are covered by observation/action parity tests.

The initial `strakam-generals-hunter:v1` policy reuses the already published
v0.2.5 game image, which contains this Hunter source and its JAX dependencies.
Its launch argv is `python -c` followed by the complete `hunter_player.py`
source. This avoids uploading the same large runtime again. The game server
does not run in the player container. The standalone image below is another
way to package the same client for subsequent versions.

Published base image:
`public.ecr.aws/q5f4m8t9/cogames@sha256:abdefc4f4174c898aaa1dea32fcd1d7bed504c6121674fc2db6ad339596c9149`.

Hunter uses JAX on CPU. It compiles all 16 supported map shapes (each side
18–21) before connecting, within the game's 180-second connection window, so
compilation does not consume a turn's 500 ms deadline. This player is for the
current classic 1v1 mode, not a general multiplayer or castle-building player.

Build from the repository root:

```sh
docker build --platform linux/amd64 --target hunter \
  -f integrations/softmax/Dockerfile -t generals-coworld-hunter:local .
coworld upload-policy generals-coworld-hunter:local --name strakam-generals-hunter \
  --run python --run -m --run integrations.softmax.hunter_player
```

The target league is
[Generals Competition](https://softmax.com/observatory/v2?detail=league:league_8c189954-be68-479c-a092-eeb79c436d12).
After checking the uploaded policy in hosted matches, submit that explicit version:

```sh
coworld submit strakam-generals-hunter:v1 \
  --league league_8c189954-be68-479c-a092-eeb79c436d12 --no-open-browser
```

Building or uploading a policy does not itself enter the competition. The
submission is a separate operation; placement can complete asynchronously.
Evidence for the initial submission is saved locally under the ignored
`integrations/softmax/local-output/hunter-submission/` directory.

Initial verification: two observation/action parity tests passed; a local
container match against Expander reached turn 1,200 with zero timeouts for
either player and a draw. The full replay loaded through its final turn in
headless Chromium without browser errors. Both hosted verification matches
(`xreq_8e07a544-1b3b-4716-b412-1867d5a11e26`) completed: Hunter beat Expander by
general capture from both seats, on turns 196 and 315, with zero timeouts for
either player. These are verification matches, not league ranking results.
The league membership is active; its first scheduled round/standing still
needs checking separately.
