# Hunter on Softmax

This player runs the existing `generals/agents/hunter_agent.py` unchanged. The
`hunter_player.py` client reconstructs the fog-limited engine observation from
the Coworld protocol and submits Hunter's actions directly over WebSocket.
Both player perspectives are covered by observation/action parity tests.

Hunter uses JAX on CPU. It compiles all 16 supported map shapes (each side
18–21) before connecting, within the game's 180-second connection window, so
compilation does not consume a turn's 500 ms deadline. This player is for the
current classic 1v1 mode, not a general multiplayer or castle-building player.

Build from the repository root:

```sh
docker build --platform linux/amd64 --target hunter \
  -f integrations/softmax/Dockerfile -t generals-coworld-hunter:local .
coworld upload-policy generals-coworld-hunter:local --name strakam-generals-hunter \
  --run 'python -m integrations.softmax.hunter_player'
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
