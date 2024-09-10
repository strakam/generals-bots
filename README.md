<div align="center">
[![Generals-Zoo logo](https://raw.githubusercontent.com/strakam/Generals-Zoo/tree/master/generals/images/test.png)](https://github.com/strakam/Generals-Zoo)
  **Machine Learning friendly implementation of the Generals.io game**
</div>
## Installation/Setup

Create python virtual environment:
```sh
python3 -m venv .venv
```

Activate the virtual environment:
```sh
source .venv/bin/activate
```

Install the requirements:
```sh
pip3 install -r requirements.txt
```

## Running the game

Run the game (and record it):
```sh
make run
```

Replay the last game:
```sh
make t_replay
```

## Controls
- `q` — quit/close the game
- `←/→` — increase/decrease the game speed
- `h/l` - to control replay frames
- `spacebar` - to pause
- `Mouse` click on the player's row — toggle the FOV (Field Of View) of the given player
